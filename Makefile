.PHONY: setup-conda setup-venv clean

# Method 1: One-click Conda environment setup (Recommended, especially for OpenCV and PyTorch)
setup-conda:
	@echo "Creating Conda environment 'comp9517'..."
	conda env create -f environment.yml
	@echo "Done! Please run: conda activate comp9517"

# Method 2: One-click Python virtual environment (venv) setup
setup-venv:
	@echo "Creating virtual environment..."
	python3 -m venv venv
	@echo "Installing dependencies..."
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "Done! Please run: source venv/bin/activate"

# Clean up environments
clean:
	rm -rf venv
