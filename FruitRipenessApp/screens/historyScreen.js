// HistoryScreen.js
// Shows a list of all past predictions fetched from the Pi.
// Refreshes automatically whenever the user opens this screen.

import React, { useState, useCallback } from 'react';
import {
  View, Text, FlatList, StyleSheet,
  TouchableOpacity, Image, ActivityIndicator
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { getHistory } from '../api/predictionApi';

export default function HistoryScreen() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  // Fetch the list of predictions from the Pi
  async function fetchHistory() {
    setLoading(true);
    try {
      const data = await getHistory();
      setHistory(data);
    } catch (err) {
      console.error('Could not load history:', err);
    } finally {
      setLoading(false);
    }
  }

  // Re-fetch every time the user navigates to this screen
  useFocusEffect(useCallback(() => { fetchHistory(); }, []));

  // Renders a single prediction card
  function renderCard({ item }) {
    const time   = new Date(item.timestamp).toLocaleString();
    const source = item.source === 'phone_camera' ? ' Phone' : ' Pi';

    return (
      <View style={styles.card}>

        {/* Photo — or a grey placeholder if there isn't one */}
        {item.image ? (
          <Image
            source={{ uri: `data:image/jpeg;base64,${item.image}` }}
            style={styles.photo}
            resizeMode="cover"
          />
        ) : (
          <View style={styles.noPhoto}>
            <Text style={styles.noPhotoText}>No image</Text>
          </View>
        )}

        {/* Label, confidence, and where/when it was taken */}
        <View style={styles.cardBody}>
          <Text style={styles.label}>{item.label}</Text>
          <Text style={styles.score}>Confidence: {(item.score * 100).toFixed(1)}%</Text>
          <Text style={styles.meta}>{source} · {time}</Text>
        </View>

      </View>
    );
  }

  return (
    <View style={styles.screen}>

      {/* Manual refresh button */}
      <TouchableOpacity style={styles.refreshButton} onPress={fetchHistory}>
        <Text style={styles.refreshText}>↻ Refresh</Text>
      </TouchableOpacity>

      {loading ? (
        <ActivityIndicator size="large" color="#2ecc71" style={{ marginTop: 40 }} />
      ) : (
        <FlatList
          data={history}
          keyExtractor={(_, index) => index.toString()}
          renderItem={renderCard}
          ListEmptyComponent={
            <Text style={styles.emptyText}>No predictions yet</Text>
          }
        />
      )}

    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 12,
  },
  refreshButton: {
    alignSelf: 'flex-end',
    marginBottom: 8,
  },
  refreshText: {
    color: '#2ecc71',
    fontWeight: 'bold',
    fontSize: 15,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    marginBottom: 14,
    overflow: 'hidden',
    elevation: 2,
  },
  photo: {
    width: '100%',
    height: 180,
  },
  noPhoto: {
    width: '100%',
    height: 100,
    backgroundColor: '#eee',
    justifyContent: 'center',
    alignItems: 'center',
  },
  noPhotoText: {
    color: '#999',
  },
  cardBody: {
    padding: 12,
  },
  label: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  score: {
    fontSize: 14,
    color: '#27ae60',
    marginTop: 2,
  },
  meta: {
    fontSize: 12,
    color: '#888',
    marginTop: 4,
  },
  emptyText: {
    textAlign: 'center',
    marginTop: 60,
    color: '#999',
    fontSize: 16,
  },
});