#!/bin/bash
mkdir -p data/raw

# Download the Cisco dataset
wget -O data/raw/cisco_network_traffic.csv "https://drive.google.com/uc?export=download&id=19trhUMgV-V475xUhbzfhNK0QnpjUeyjF"
