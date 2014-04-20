#!/bin/bash

set -e
# Any subsequent commands which fail will cause the shell script to exit immediately

# Clean previous install
ant uninstall
ant clean