# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gi.repository import GLib, GObject, Gst, GstBase, GstVideo, Gtk
import gi
import numpy as np
import sys
import threading
import time


gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('Gtk', '3.0')

Gst.init(None)

class GstPipeline:
    def __init__(self, pipeline, inf_callback, render_callback, src_size):
        self.inf_callback = inf_callback
        self.render_callback = render_callback
        self.running = False
        self.gstbuffer = None
        self.output = None
        self.sink_size = None
        self.src_size = src_size
        self.box = None
        self.condition = threading.Condition()

        self.pipeline = Gst.parse_launch(pipeline)
        self.freezer = self.pipeline.get_by_name('freezer')
        self.overlay = self.pipeline.get_by_name('overlay')
        self.overlaysink = self.pipeline.get_by_name('overlaysink')
        appsink = self.pipeline.get_by_name('appsink')
        appsink.connect('new-sample', self.on_new_sample)

        # Set up a pipeline bus watch to catch errors.
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self.on_bus_message)

        # Set up a full screen window on Coral, no-op otherwise.
        self.setup_window()

    def run(self):
        # Start inference worker.
        self.running = True
        inf_worker = threading.Thread(target=self.inference_loop)
        inf_worker.start()
        render_worker = threading.Thread(target=self.render_loop)
        render_worker.start()

        # Run pipeline.
        self.pipeline.set_state(Gst.State.PLAYING)
        self.pipeline.get_state(Gst.CLOCK_TIME_NONE)

        # We're high latency on higher resolutions, don't drop our late frames.
        if self.overlaysink:
            sinkelement = self.overlaysink.get_by_interface(GstVideo.VideoOverlay)
        else:
            sinkelement = self.pipeline.get_by_interface(GstVideo.VideoOverlay)
        sinkelement.set_property('sync', False)
        sinkelement.set_property('qos', False)

        try:
            Gtk.main()
        except:
            pass

        # Clean up.
        self.pipeline.set_state(Gst.State.NULL)
        while GLib.MainContext.default().iteration(False):
            pass
        with self.condition:
            self.running = False
            self.condition.notify_all()
        inf_worker.join()
        render_worker.join()

    def on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            Gtk.main_quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write('Warning: %s: %s\n' % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write('Error: %s: %s\n' % (err, debug))
            Gtk.main_quit()
        return True

    def on_new_sample(self, sink):
        sample = sink.emit('pull-sample')
        if not self.sink_size:
            s = sample.get_caps().get_structure(0)
            self.sink_size = (s.get_value('width'), s.get_value('height'))
        with self.condition:
            self.gstbuffer = sample.get_buffer()
            self.condition.notify_all()
        return Gst.FlowReturn.OK

    def get_box(self):
        if not self.box:
            glbox = self.pipeline.get_by_name('glbox')
            if glbox:
                glbox = glbox.get_by_name('filter')
            box = self.pipeline.get_by_name('box')
            assert glbox or box
            assert self.sink_size
            if glbox:
                self.box = (glbox.get_property('x'), glbox.get_property('y'),
                        glbox.get_property('width'), glbox.get_property('height'))
            else:
                self.box = (-box.get_property('left'), -box.get_property('top'),
                    self.sink_size[0] + box.get_property('left') + box.get_property('right'),
                    self.sink_size[1] + box.get_property('top') + box.get_property('bottom'))
        return self.box

    def inference_loop(self):
        while True:
            with self.condition:
                while not self.gstbuffer and self.running:
                    self.condition.wait()
                if not self.running:
                    break
                gstbuffer = self.gstbuffer
                self.gstbuffer = None

            # Input tensor is expected to be tightly packed, that is,
            # width and stride in pixels are expected to be the same.
            # For the Coral devboard using GPU this will always be true,
            # but when using generic GStreamer CPU based elements the line
            # stride will always be a multiple of 4 bytes in RGB format.
            # In case of mismatch we have to copy the input line by line.
            # For best performance input tensor size should take this
            # into account when using CPU based elements.
            # TODO: Use padded posenet models to avoid this.
            meta = GstVideo.buffer_get_video_meta(gstbuffer)
            assert meta and meta.n_planes == 1
            bpp = 3 # bytes per pixel.
            buf_stride = meta.stride[0] # 0 for first and only plane.
            inf_stride = meta.width * bpp

            if inf_stride == buf_stride:
                # Fast case, pass buffer as input tensor as is.
                input_tensor = gstbuffer
            else:
                # Slow case, need to pack lines tightly (copy).
                result, mapinfo = gstbuffer.map(Gst.MapFlags.READ)
                assert result
                data_view = memoryview(mapinfo.data)
                input_tensor = bytearray(inf_stride * meta.height)
                src_offset = dst_offset = 0
                for row in range(meta.height):
                    src_end = src_offset + inf_stride
                    dst_end = dst_offset + inf_stride
                    input_tensor[dst_offset : dst_end] = data_view[src_offset : src_end]
                    src_offset += buf_stride
                    dst_offset += inf_stride
                input_tensor = bytes(input_tensor)
                gstbuffer.unmap(mapinfo)

            output = self.inf_callback(input_tensor)
            with self.condition:
                self.output = output
                self.condition.notify_all()

    def render_loop(self):
        while True:
            with self.condition:
                while not self.output and self.running:
                    self.condition.wait()
                if not self.running:
                    break
                output = self.output
                self.output = None

            svg, freeze = self.render_callback(output, self.src_size, self.get_box())
            self.freezer.frozen = freeze
            if self.overlaysink:
                self.overlaysink.set_property('svg', svg)
            elif self.overlay:
                self.overlay.set_property('data', svg)

    def setup_window(self):
        # Only set up our own window if we have Coral overlay sink in the pipeline.
        if not self.overlaysink:
            return

        gi.require_version('GstGL', '1.0')
        from gi.repository import GstGL

        # Needed to commit the wayland sub-surface.
        def on_gl_draw(sink, widget):
            widget.queue_draw()

        # Needed to account for window chrome etc.
        def on_widget_configure(widget, event, overlaysink):
            allocation = widget.get_allocation()
            overlaysink.set_render_rectangle(allocation.x, allocation.y,
                    allocation.width, allocation.height)
            return False

        window = Gtk.Window(Gtk.WindowType.TOPLEVEL)
        window.fullscreen()

        drawing_area = Gtk.DrawingArea()
        window.add(drawing_area)
        drawing_area.realize()

        self.overlaysink.connect('drawn', on_gl_draw, drawing_area)

        # Wayland window handle.
        wl_handle = self.overlaysink.get_wayland_window_handle(drawing_area)
        self.overlaysink.set_window_handle(wl_handle)

        # Wayland display context wrapped as a GStreamer context.
        wl_display = self.overlaysink.get_default_wayland_display_context()
        self.overlaysink.set_context(wl_display)

        drawing_area.connect('configure-event', on_widget_configure, self.overlaysink)
        window.connect('delete-event', Gtk.main_quit)
        window.show_all()

        # The appsink pipeline branch must use the same GL display as the screen
        # rendering so they get the same GL context. This isn't automatically handled
        # by GStreamer as we're the ones setting an external display handle.
        def on_bus_message_sync(bus, message, overlaysink):
            if message.type == Gst.MessageType.NEED_CONTEXT:
                _, context_type = message.parse_context_type()
                if context_type == GstGL.GL_DISPLAY_CONTEXT_TYPE:
                    sinkelement = overlaysink.get_by_interface(GstVideo.VideoOverlay)
                    gl_context = sinkelement.get_property('context')
                    if gl_context:
                        display_context = Gst.Context.new(GstGL.GL_DISPLAY_CONTEXT_TYPE, True)
                        GstGL.context_set_gl_display(display_context, gl_context.get_display())
                        message.src.set_context(display_context)
            return Gst.BusSyncReply.PASS

        bus = self.pipeline.get_bus()
        bus.set_sync_handler(on_bus_message_sync, self.overlaysink)

def on_bus_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write('Warning: %s: %s\n' % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write('Error: %s: %s\n' % (err, debug))
        loop.quit()
    return True

def detectCoralDevBoard():
    try:
        if 'MX8MQ' in open('/sys/firmware/devicetree/base/model').read():
            print('Detected Edge TPU dev board.')
            return True
    except:
        pass
    return False

class Freezer(GstBase.BaseTransform):
    __gstmetadata__ = ('<longname>', '<class>', '<description>', '<author>')
    __gsttemplates__ = (Gst.PadTemplate.new('sink',
                            Gst.PadDirection.SINK,
                            Gst.PadPresence.ALWAYS,
                            Gst.Caps.new_any()),
                        Gst.PadTemplate.new('src',
                            Gst.PadDirection.SRC,
                            Gst.PadPresence.ALWAYS,
                            Gst.Caps.new_any())
                        )
    def __init__(self):
        self.buf = None
        self.frozen = False
        self.set_passthrough(False)

    def do_prepare_output_buffer(self, inbuf):
        if self.frozen:
            if not self.buf:
                self.buf = inbuf
            src_buf = self.buf
        else:
            src_buf = inbuf
        buf = Gst.Buffer.new()
        buf.copy_into(src_buf, Gst.BufferCopyFlags.FLAGS | Gst.BufferCopyFlags.TIMESTAMPS |
            Gst.BufferCopyFlags.META | Gst.BufferCopyFlags.MEMORY, 0, inbuf.get_size())
        buf.pts = inbuf.pts

        return (Gst.FlowReturn.OK, buf)

    def do_transform(self, inbuf, outbuf):
        return Gst.FlowReturn.OK

def register_elements(plugin):
    gtype = GObject.type_register(Freezer)
    Gst.Element.register(plugin, 'freezer', 0, gtype)
    return True

Gst.Plugin.register_static(
    Gst.version()[0], Gst.version()[1], # GStreamer version
    '',                                 # name
    '',                                 # description
    register_elements,                  # init_func
    '',                                 # version
    'unknown',                          # license
    '',                                 # source
    '',                                 # package
    ''                                  # origin
)

def run_pipeline(inf_callback, render_callback, src_size,
                 inference_size,
                 mirror=False,
                 h264=False,
                 jpeg=False,
                 videosrc='/dev/video0'):
    if h264:
        SRC_CAPS = 'video/x-h264,width={width},height={height},framerate=30/1'
    elif jpeg:
        SRC_CAPS = 'image/jpeg,width={width},height={height},framerate=30/1'
    else:
        SRC_CAPS = 'video/x-raw,width={width},height={height},framerate=30/1'
    PIPELINE = 'v4l2src device=%s ! {src_caps}' % videosrc

    scale = min(inference_size[0] / src_size[0],
                inference_size[1] / src_size[1])
    scale = tuple(int(x * scale) for x in src_size)
    scale_caps = 'video/x-raw,width={width},height={height}'.format(
        width=scale[0], height=scale[1])
    PIPELINE += """ ! decodebin ! videoflip video-direction={direction} ! tee name=t
               t. ! {leaky_q} ! videoconvert ! freezer name=freezer ! rsvgoverlay name=overlay
                  ! videoconvert ! autovideosink
               t. ! {leaky_q} ! videoconvert ! videoscale ! {scale_caps} ! videobox name=box autocrop=true
                  ! {sink_caps} ! {sink_element}
            """

    #TODO: Fix pipeline for the dev board.
    SINK_ELEMENT = 'appsink name=appsink emit-signals=true max-buffers=1 drop=true'
    SINK_CAPS = 'video/x-raw,format=RGB,width={width},height={height}'
    LEAKY_Q = 'queue max-size-buffers=1 leaky=downstream'
    direction = 'horiz' if mirror else 'identity'

    src_caps = SRC_CAPS.format(width=src_size[0], height=src_size[1])
    sink_caps = SINK_CAPS.format(width=inference_size[0], height=inference_size[1])
    pipeline = PIPELINE.format(src_caps=src_caps, sink_caps=sink_caps,
        sink_element=SINK_ELEMENT, direction=direction, leaky_q=LEAKY_Q, scale_caps=scale_caps)
    print('Gstreamer pipeline: ', pipeline)
    pipeline = GstPipeline(pipeline, inf_callback, render_callback, src_size)
    pipeline.run()
