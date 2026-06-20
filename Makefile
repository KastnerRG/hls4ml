IMAGE ?= hls4ml-vitis:2025.2
CONTAINER ?= hls4ml-vitis-$(shell id -u)
DOCKER ?= docker
VITIS_ROOT ?= /tools/Xilinx/Vivado/2025.2
DOCNAV_ROOT ?= $(abspath $(VITIS_ROOT)/../DocNav)
LICENSE_DIR ?= $(HOME)/Xilinx-lic
LICENSE_SERVER ?= 2100@cselm2.ucsd.edu
RF ?= 16
HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)
HOST_USER := $(shell id -un)
IDENTITY_DIR := $(HOME)/.cache/hls4ml-docker
DISPLAY ?= $(shell printenv DISPLAY)
XAUTHORITY ?= $(shell printenv XAUTHORITY)

X11_ARGS := $(if $(DISPLAY),--env DISPLAY=$(DISPLAY) --env QT_X11_NO_MITSHM=1 --volume /tmp/.X11-unix:/tmp/.X11-unix:rw $(if $(wildcard $(XAUTHORITY)),--env XAUTHORITY=/tmp/.Xauthority --volume $(XAUTHORITY):/tmp/.Xauthority:ro))

.PHONY: build start enter stop jet-simple

build:
	$(DOCKER) build --tag $(IMAGE) .

start:
	@test -d "$(VITIS_ROOT)" || { echo "Vitis not found at $(VITIS_ROOT)" >&2; exit 1; }
	@test -d "$(DOCNAV_ROOT)" || { echo "DocNav not found at $(DOCNAV_ROOT)" >&2; exit 1; }
	@test -d "$(LICENSE_DIR)" || { echo "License directory not found at $(LICENSE_DIR)" >&2; exit 1; }
	@$(DOCKER) image inspect $(IMAGE) >/dev/null 2>&1 || $(MAKE) build
	@if $(DOCKER) container inspect $(CONTAINER) >/dev/null 2>&1 \
		&& [ "$$($(DOCKER) inspect -f '{{.Image}}' $(CONTAINER))" != "$$($(DOCKER) image inspect -f '{{.Id}}' $(IMAGE))" ]; then \
		$(DOCKER) rm -f $(CONTAINER); \
	fi
	@mkdir -p "$(HOME)/.Xilinx" "$(IDENTITY_DIR)"
	@awk -F: -v uid="$(HOST_UID)" '$$3 != uid { print }' /etc/passwd > "$(IDENTITY_DIR)/passwd"
	@printf '%s:x:%s:%s:Container user:/home/hls4ml:/bin/bash\n' "$(HOST_USER)" "$(HOST_UID)" "$(HOST_GID)" >> "$(IDENTITY_DIR)/passwd"
	@awk -F: -v gid="$(HOST_GID)" '$$3 != gid { print }' /etc/group > "$(IDENTITY_DIR)/group"
	@printf '%s:x:%s:\n' "$(HOST_USER)" "$(HOST_GID)" >> "$(IDENTITY_DIR)/group"
	@if ! $(DOCKER) container inspect $(CONTAINER) >/dev/null 2>&1; then \
		$(DOCKER) run --detach --name $(CONTAINER) \
			--init --user $(HOST_UID):$(HOST_GID) --hostname $(CONTAINER) --network host --ipc host --shm-size 8g \
			--workdir /workspace --env HOME=/home/hls4ml --env HLS4ML_VITIS_ROOT=$(VITIS_ROOT) \
			--env XILINXD_LICENSE_FILE=$(LICENSE_SERVER) --env LM_LICENSE_FILE=$(LICENSE_SERVER) \
			--volume $(CURDIR):/workspace --volume $(VITIS_ROOT):$(VITIS_ROOT):ro \
			--volume $(DOCNAV_ROOT):$(DOCNAV_ROOT):ro \
			--volume $(LICENSE_DIR):/licenses/Xilinx-lic:ro \
			--volume $(HOME)/.Xilinx:/home/hls4ml/.Xilinx \
			--volume $(HOME)/.cache/hls4ml-docker:/home/hls4ml/.cache \
			--volume $(IDENTITY_DIR)/passwd:/etc/passwd:ro --volume $(IDENTITY_DIR)/group:/etc/group:ro \
			$(X11_ARGS) $(IMAGE) >/dev/null; \
	elif [ "$$($(DOCKER) inspect -f '{{.State.Running}}' $(CONTAINER))" != true ]; then \
		$(DOCKER) start $(CONTAINER) >/dev/null; \
	fi

enter: start
	@printf 'Entering Docker container %s as %s (uid=%s gid=%s)\n' "$(CONTAINER)" "$(HOST_USER)" "$(HOST_UID)" "$(HOST_GID)"
	@$(DOCKER) exec -it $(CONTAINER) /usr/local/bin/hls4ml-shell

jet-simple: start
	@printf 'Running jet simple in %s with reuse factor %s\n' "$(CONTAINER)" "$(RF)"
	@$(DOCKER) exec $(CONTAINER) /usr/local/bin/hls4ml-shell python nn_exp_simple/jet.py --reuse-factor $(RF)

stop:
	-$(DOCKER) rm -f $(CONTAINER)
