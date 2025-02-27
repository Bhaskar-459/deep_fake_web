#!/bin/bash

# Fetch Git LFS files
git lfs pull

# Start the Flask app using Gunicorn
gunicorn app:app --bind 0.0.0.0:$PORT