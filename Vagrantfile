# -*- mode: ruby -*-
# vi: set ft=ruby :

# All Vagrant configuration is done below. The "2" in Vagrant.configure
# configures the configuration version (we support older styles for
# backwards compatibility). Please don't change it unless you know what
# you're doing.
Vagrant.configure("2") do |config|
  # The most common configuration options are documented and commented below.
  # For a complete reference, please see the online documentation at
  # https://docs.vagrantup.com.

  # Every Vagrant development environment requires a box. You can search for
  # boxes at https://vagrantcloud.com/search.
  config.vm.box = "generic/ubuntu1804"

  # Booting takes 5-10 minutes for ubuntu official boxes, but the `generic`
  # ones are much faster. I'll leave this as 10 minutes in case anyone wants
  # to switch to a different config.vm.box later.
  config.vm.boot_timeout = 600

  config.vm.provider "virtualbox" do |vb|
    # This requires `brew cask install virtualbox-extension-pack` for USB3
    vb.customize ["modifyvm", :id, "--usbxhci", "on"]

    # There doesn't seem to be a way to idempotently add a rule, and any command
    # that fails will abort vm startup, so there's no way for me to clean up
    # old rules reliably either.
    # This means that you will get duplicate rules with the same name piling up
    # in the ui, but you'll only notice them if you go looking. I'm very sorry.

    # This is the first device that shows up on the bus.
    vb.customize [
      'usbfilter', 'add', '0',
        '--target', :id,
        '--name', 'Coral USB Accelerator',
        '--vendorid', '0x1a6e',
        '--productid', '0x089a',
    ]
    # This is the second device that shows up when you actually try to use the
    # thing from a python script.
    # See https://dev.to/kojikanao/coral-edgetpu-usb-with-virtualbox-57e1 for
    # details.
    vb.customize [
      'usbfilter', 'add', '0',
        '--target', :id,
        '--name', 'Google Inc. Coral USB Accelerator',
        '--vendorid', '0x18d1',
        '--productid', '0x9302',
    ]

    # # Customize the amount of memory on the VM:
    # vb.memory = "1024"
  end

  config.vm.provision "shell", inline: <<-SHELL
    # lsusb is useful for debugging the coral usb forwarding:
    apt install -y usbutils
    # from https://coral.withgoogle.com/docs/accelerator/get-started/
    wget https://dl.google.com/coral/edgetpu_api/edgetpu_api_latest.tar.gz \
        -O edgetpu_api.tar.gz --trust-server-names
    tar xzf edgetpu_api.tar.gz
    # TODO: there is a line in the script that says:
    #     sudo udevadm control --reload-rules && udevadm trigger
    # when it should say:
    #     sudo udevadm control --reload-rules && sudo udevadm trigger
    # As a work-around, run the whole thing as root :-(
    yes | sudo edgetpu_api/install.sh
  SHELL
  # Disable automatic box update checking. If you disable this, then
  # boxes will only be checked for updates when the user runs
  # `vagrant box outdated`. This is not recommended.
  # config.vm.box_check_update = false

  # Create a forwarded port mapping which allows access to a specific port
  # within the machine from a port on the host machine. In the example below,
  # accessing "localhost:8080" will access port 80 on the guest machine.
  # NOTE: This will enable public access to the opened port
  # config.vm.network "forwarded_port", guest: 80, host: 8080

  # Create a forwarded port mapping which allows access to a specific port
  # within the machine from a port on the host machine and only allow access
  # via 127.0.0.1 to disable public access
  # config.vm.network "forwarded_port", guest: 80, host: 8080, host_ip: "127.0.0.1"

  # Create a private network, which allows host-only access to the machine
  # using a specific IP.
  # config.vm.network "private_network", ip: "192.168.33.10"

  # Create a public network, which generally matched to bridged network.
  # Bridged networks make the machine appear as another physical device on
  # your network.
  # config.vm.network "public_network"

  # Share an additional folder to the guest VM. The first argument is
  # the path on the host to the actual folder. The second argument is
  # the path on the guest to mount the folder. And the optional third
  # argument is a set of non-required options.
  config.vm.synced_folder ".", "/home/vagrant/project-posenet"
end
