for running the file ,

make sure your port is not being used on your system — most likely by:

a previous Streamlit app

or

a previous Docker container still running in the background

docker run -p 8501:8501 ednaai:latest
