#!/bin/bash
mkdir -p data/raw

# Download the Cisco dataset
wget -O data/raw/cisco_network_traffic.csv "https://drive.google.com/uc?export=download&id=19trhUMgV-V475xUhbzfhNK0QnpjUeyjF"

# Download the NERC dataset
wget -O data/raw/nerc_flow.csv "https://drive.google.com/file/d/1fKd8QbCMumCjZgppxodCAy6Sw44BUhm3/view?usp=drive_link"

