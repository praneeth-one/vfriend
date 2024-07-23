import os
import re
import io
import cv2
import boto3
import base64
import time
from flask import Flask, request, redirect, url_for, render_template, flash, jsonify

app = Flask(__name__)
app.secret_key = "supersecretkey"  #you don't need to fill this

# Your AWS credentials
AWS_ACCESS_KEY_ID = 'IAM Access Key'
AWS_SECRET_ACCESS_KEY = 'IAM secret Key'
AWS_REGION = 'Region your bucket in' 
LEX_BOT_NAME = 'YourBotName'
LEX_BOT_ALIAS = 'YourBotAlias'
USER_ID = 'unique_user_id'

S3_BUCKET_NAME = 'your-s3-bucket-name'
#TARGET_IMAGES = ['target-image1.jpg', 'target-image2.jpg']  # Add more images as needed

lex_client = boto3.client(
    'lex-runtime',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
polly_client = boto3.client(
    'polly',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
rekognition_client = boto3.client(
    'rekognition', 
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
transcribe_client = boto3.client(
    'transcribe', 
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
s3_client = boto3.client('s3', region_name=AWS_REGION)

def compare_faces(source_image_data):
    for target_image in get_images_from_bucket():
        response = rekognition_client.compare_faces(
            SourceImage={'Bytes': source_image_data},
            TargetImage={'S3Object': {'Bucket': S3_BUCKET_NAME, 'Name': target_image}},
            SimilarityThreshold=90
        )
        if response['FaceMatches']:
            return True
    return False

def get_images_from_bucket():
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME )
        return [content['Key'] for content in response.get('Contents', [])]
    except Exception as e:
        print(f"Error listing images in bucket {S3_BUCKET_NAME }: {e}")
        flash(f"Error listing images in bucket {S3_BUCKET_NAME }: {e}", 'danger')
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify_face', methods=['POST'])
def verify_face():
    image_data = request.json.get('image_data')
    if not image_data:
        return jsonify({'error': 'No image provided'}), 400

    image_bytes = base64.b64decode(image_data.split(',')[1])

    if compare_faces(image_bytes):
        return jsonify({'verified': True}), 200
    else:
        return jsonify({'verified': False}), 403

@app.route('/send_message', methods=['POST'])
def send_message():
    audio_data = request.json.get('audio_data')
    if not audio_data:
        return jsonify({'error': 'No audio provided'}), 400

    # Save the audio data to a temporary file
    audio_bytes = base64.b64decode(audio_data.split(',')[1])
    audio_file_name = f'temp_audio_{int(time.time())}.wav'
    with open(audio_file_name, 'wb') as audio_file:
        audio_file.write(audio_bytes)

    # Upload the audio file to S3 for transcription
    s3_client.upload_file(audio_file_name, S3_BUCKET_NAME, audio_file_name)

    # Start the transcription job
    transcribe_job_name = f'transcription_job_{int(time.time())}'
    transcribe_client.start_transcription_job(
        TranscriptionJobName=transcribe_job_name,
        Media={'MediaFileUri': f's3://{S3_BUCKET_NAME}/{audio_file_name}'},
        MediaFormat='wav',
        LanguageCode='en-US'
    )

    # Wait for the transcription job to complete
    while True:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=transcribe_job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(5)

    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'FAILED':
        return jsonify({'error': 'Transcription failed'}), 500

    # Get the transcribed text
    transcription_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
    transcription_text = re.get(transcription_url).json()['results']['transcripts'][0]['transcript']

    # Send the transcribed text to Lex
    lex_response = lex_client.post_text(
        botName=LEX_BOT_NAME,
        botAlias=LEX_BOT_ALIAS,
        userId=USER_ID,
        inputText=transcription_text
    )

    bot_message = lex_response.get('message', 'I didn\'t understand that. Can you please rephrase?')

    # Convert bot's response to speech using Polly
    polly_response = polly_client.synthesize_speech(
        Text=bot_message,
        OutputFormat='mp3',
        VoiceId='Joanna'
    )

    audio_stream = io.BytesIO(polly_response['AudioStream'].read())

    return jsonify({
        'bot_message': bot_message,
        'audio_content': base64.b64encode(audio_stream.getvalue()).decode('ISO-8859-1')
    })

if __name__ == "__main__":
    app.run(debug=True)
