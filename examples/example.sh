#!/bin/bash

upload_url="http://localhost:8181/upload"
ocr_url="http://localhost:8181/ocr"
extract_url="http://localhost:8181/extract"

file1="東京都建築安全条例.pdf"
file2="建築基準法施行令.pdf"

query="第三条の二において、路地状部分の幅員が4メートル未満の敷地に建築してはならない建築物の階数について、耐火建築物、準耐火建築物または知事が定めた構造方法を用いる建築物の場合、いくつまでの階数が認められていますか。"

# Upload files
upload_response=$(curl -s -X POST -F "files=@$file1" -F "files=@$file2" "$upload_url")

# Extract file_id and signed_url from the upload response
file_id=$(echo "$upload_response" | jq -r '.data[0].file_id')
signed_url=$(echo "$upload_response" | jq -r '.data[0].signed_url')

# Perform OCR
ocr_data='{"signed_url":"'"$signed_url"'"}'
ocr_response=$(curl -s -X POST -H "Content-Type: application/json" -d "$ocr_data" "$ocr_url")

# Extract task_id from the OCR response
task_id=$(echo "$ocr_response" | jq -r '.data.task_id')

# Wait for OCR processing to complete
while true; do
  extract_data='{"file_hash":"'"$task_id"'", "query": "'"$query"'"}'
  extract_response=$(curl -s -X POST -H "Content-Type: application/json" -d "$extract_data" "$extract_url")
  
  status=$(echo "$extract_response" | jq -r '.status')
  
  if [ "$status" = "success" ]; then
    answer=$(echo "$extract_response" | jq -r '.answer.answer')
    echo "Answer: $answer"
    break
  fi
  
  echo "Processing..."
  sleep 1
done
