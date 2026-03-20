// CaptureScreen.js
// Lets the user take a photo and get a ripeness prediction.
// Option 1: The Raspberry Pi camera captures and predicts on its own.
// Option 2: The phone takes a photo, sends it to the Pi to predict.

import React, { useState } from 'react';
import {
  View, Text, TouchableOpacity, Image,
  ActivityIndicator, StyleSheet, Alert, ScrollView
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { captureWithPiCamera, predictWithPhonePhoto } from '../api/predictionApi';

export default function CaptureScreen() {
  const [photo, setPhoto]     = useState(null);   // URI of the phone photo preview
  const [result, setResult]   = useState(null);   // { label, score, source }
  const [loading, setLoading] = useState(false);

  // --- Option 1: Pi camera does everything ---
  async function usePiCamera() {
    setLoading(true);
    setResult(null);
    setPhoto(null);
    try {
      const data = await captureWithPiCamera();
      setResult(data);
    } catch (err) {
      Alert.alert('Error', `Could not reach the Pi: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  // --- Option 2: Phone takes photo, Pi predicts ---
  async function usePhoneCamera() {
    // Ask for camera permission first
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Please allow camera access.');
      return;
    }

    // Open the camera
    const picked = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
      quality: 0.7,
    });

    if (picked.canceled) return;

    const uri = picked.assets[0].uri;
    setPhoto(uri);
    setResult(null);
    setLoading(true);

    try {
      const data = await predictWithPhonePhoto(uri);
      setResult(data);
    } catch (err) {
      Alert.alert('Error', `Prediction failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  // Pick a border colour for the result card based on ripeness
  function cardColour(label = '') {
    const l = label.toLowerCase();
    if (l.includes('unripe')) return '#f57f17'; // orange = not ready
    if (l.includes('ripe'))   return '#2e7d32'; // green  = ready
    if (l.includes('over'))   return '#b71c1c'; // red    = too late
    return '#333';
  }

  return (
    <ScrollView contentContainerStyle={styles.screen}>

      <Text style={styles.heading}>How do you want to take the photo?</Text>

      {/* Button: use the Pi camera */}
      <TouchableOpacity
        style={styles.greenButton}
        onPress={usePiCamera}
        disabled={loading}
      >
        <Text style={styles.buttonTitle}>🍓 Use Pi Camera</Text>
        <Text style={styles.buttonSub}>Captures directly from the Raspberry Pi</Text>
      </TouchableOpacity>

      {/* Button: use the phone camera */}
      <TouchableOpacity
        style={styles.blueButton}
        onPress={usePhoneCamera}
        disabled={loading}
      >
        <Text style={styles.buttonTitle}>📱 Use Phone Camera</Text>
        <Text style={styles.buttonSub}>Takes photo here, sends to Pi for prediction</Text>
      </TouchableOpacity>

      {/* Loading spinner */}
      {loading && (
        <ActivityIndicator size="large" style={{ marginTop: 24 }} />
      )}

      {/* Phone photo preview */}
      {photo && (
        <Image source={{ uri: photo }} style={styles.preview} />
      )}

      {/* Result card */}
      {result && (
        <View style={[styles.resultCard, { borderColor: cardColour(result.label) }]}>
          <Text style={styles.resultLabel}>{result.label}</Text>
          <Text style={styles.resultScore}>
            Confidence: {(result.score * 100).toFixed(1)}%
          </Text>
          <Text style={styles.resultSource}>Source: {result.source}</Text>
        </View>
      )}

    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: {
    padding: 24,
    alignItems: 'center',
    backgroundColor: '#f9f9f9',
    flexGrow: 1,
  },
  heading: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 24,
  },
  greenButton: {
    backgroundColor: '#2e7d32',
    padding: 18,
    borderRadius: 10,
    width: '100%',
    alignItems: 'center',
    marginBottom: 16,
  },
  blueButton: {
    backgroundColor: '#1565c0',
    padding: 18,
    borderRadius: 10,
    width: '100%',
    alignItems: 'center',
  },
  buttonTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  buttonSub: {
    color: '#ddd',
    fontSize: 12,
    marginTop: 4,
  },
  preview: {
    width: 280,
    height: 210,
    borderRadius: 10,
    marginTop: 20,
  },
  resultCard: {
    marginTop: 24,
    padding: 20,
    borderWidth: 3,
    borderRadius: 12,
    width: '100%',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  resultLabel: {
    fontSize: 24,
    fontWeight: '700',
    marginBottom: 6,
  },
  resultScore: {
    fontSize: 16,
    color: '#444',
  },
  resultSource: {
    fontSize: 13,
    color: '#888',
    marginTop: 4,
  },
});