// HomeScreen.js
// The first thing the user sees.
// Two buttons: go make a prediction, or look at past predictions.
 
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
 
export default function HomeScreen({ navigation }) {
  return (
    <View style={styles.screen}>
 
      <Text style={styles.title}> Fruit Ripeness</Text>
      <Text style={styles.subtitle}>Check if your fruit is ready to eat</Text>
 
      <TouchableOpacity
        style={styles.greenButton}
        onPress={() => navigation.navigate('Capture')}
      >
        <Text style={styles.buttonText}>📸 New Prediction</Text>
      </TouchableOpacity>
 
      <TouchableOpacity
        style={styles.blueButton}
        onPress={() => navigation.navigate('History')}
      >
        <Text style={styles.buttonText}>📋 View History</Text>
      </TouchableOpacity>
 
    </View>
  );
}
 
const styles = StyleSheet.create({
  screen: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
    backgroundColor: '#f9f9f9',
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 15,
    color: '#666',
    marginBottom: 48,
  },
  greenButton: {
    backgroundColor: '#2e7d32',
    padding: 16,
    borderRadius: 10,
    width: '100%',
    alignItems: 'center',
    marginBottom: 16,
  },
  blueButton: {
    backgroundColor: '#1565c0',
    padding: 16,
    borderRadius: 10,
    width: '100%',
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});