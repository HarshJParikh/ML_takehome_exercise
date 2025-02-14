# Use the official Python 3.10 image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install compatible versions of `huggingface_hub` and `transformers`
RUN pip install --no-cache-dir huggingface_hub==0.14.1 transformers==4.30.0

# Copy the dataset CSV files into the container
COPY Unique_Fixed_Sentiment_Analysis_Dataset.csv .
COPY Unique_Intent_Classification_Dataset.csv .

# Copy the Python script into the container
COPY ml_takehome_exercise.py .

# Set the command to run the script
CMD ["python", "ml_takehome_exercise.py"]
