import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import HomeScreen    from './screens/homeScreen';
import CaptureScreen from './screens/captureScreen';
import HistoryScreen from './screens/historyScreen';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen name="Home"    component={HomeScreen}    options={{ title: ' Fruit Ripeness' }} />
        <Stack.Screen name="Capture" component={CaptureScreen} options={{ title: 'New Prediction' }} />
        <Stack.Screen name="History" component={HistoryScreen} options={{ title: 'History' }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
