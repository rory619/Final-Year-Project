// components/ImagePickerButton.js
// A button that opens the phone camera and shoows a result.
//   image: current photo URI (or null if none taken yet)
//   onChange — called with the new photo URI after the user takes a shot

//   disabled — set to true while a request is being made

import React from 'react';
import { View, Text, TouchableOpacity, Image, StyleSheet, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

export default function ImagePickerButton({ image, onChange, disabled }) {

  async function openCamera() {
    //  permission
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Please allow camera access.');
      return;
    }

    // Open  camera
    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
      quality: 0.7,
    });

    if (!result.canceled) {
      onChange(result.assets[0].uri);
    }
  }

  return (
    <View style={styles.row}>

      <TouchableOpacity
        style={[styles.button, disabled && styles.buttonDisabled]}
        onPress={openCamera}
        disabled={disabled}
      >
        <Text style={styles.buttonText}>
          {image ? '📷 Retake Photo' : '📷 Take Photo'}
        </Text>
      </TouchableOpacity>

      {/* Show thumbnail once a photo has been taken */}
      {image && (
        <Image source={{ uri: image }} style={styles.thumbnail} />
      )}

    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 8,
  },
  button: {
    backgroundColor: '#2e7d32',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 14,
  },
  thumbnail: {
    width: 80,
    height: 80,
    borderRadius: 6,
    marginLeft: 12,
  },
});