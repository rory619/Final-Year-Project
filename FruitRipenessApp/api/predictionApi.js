// api/predictionApi.js
// All network calls to the Raspberry Pi live here.
// Import PI_URL from config — it should look like "http://192.168.x.x:5000"

import { PI_URL } from '../config';

// Tell the Pi camera to capture a photo and predict ripeness.
// Returns: { label, score, source, timestamp }
export async function captureWithPiCamera() {
  const res = await fetch(`${PI_URL}/capture`);
  if (!res.ok) throw new Error(`Pi returned status ${res.status}`);
  return res.json();
}

// Send a photo taken on the phone to the Pi for prediction.
// uri     — local file URI from expo-image-picker
// Returns: { label, score, source, timestamp }
export async function predictWithPhonePhoto(uri) {
  const form = new FormData();
  form.append('file', {
    uri,
    name: 'photo.jpg',
    type: 'image/jpeg',
  });

  const res = await fetch(`${PI_URL}/predict`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error(`Pi returned status ${res.status}`);
  return res.json();
}

// Fetch all past predictions stored on the Pi.
// Returns: array of { label, score, source, timestamp, image? }
export async function getHistory() {
  const res = await fetch(`${PI_URL}/history`);
  if (!res.ok) throw new Error(`Pi returned status ${res.status}`);
  return res.json();
}