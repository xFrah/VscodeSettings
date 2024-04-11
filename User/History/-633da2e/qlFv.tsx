// Disclaimer.tsx
import React from 'react';
import { useDisclaimer } from './disclaimerContext';
import { View, StyleSheet, Text } from 'react-native';
import { Button } from 'react-native-paper';
import { useAuth } from '../views/AuthContext';
import { black } from 'react-native-paper/lib/typescript/src/styles/themes/v2/colors';

const Disclaimer = () => {
    const { disclaimerAccepted, handleAcceptDisclaimer, profileId } = useDisclaimer();
    const { isAuthenticated } = useAuth();

    if (!isAuthenticated) {
        return null;
    }
    console.log(disclaimerAccepted," acce");
    if (disclaimerAccepted) {
        return null;
    }

    const styles = StyleSheet.create({
        smallRectangle: {
            position: 'absolute',
            bottom: 70,
            right: 0,
            width: 290,
            height: 110,
            backgroundColor: '#171B27',
            borderColor: 'white',
            borderWidth: 1,
            borderRadius: 6,
            marginBottom: 20,
            marginRight: 20,
            alignItems: 'stretch',
            opacity: 0.7,
        },
        button: {
            paddingTop: 3,
            bottom: '0',
            backgroundColor: '#f2a007', // Colore di sfondo del bottone
            border: 'none',
            color: 'white',
            padding: 10,
            fontSize: 16,
            fontWeight: 'bold',
            textAlign: 'center',
            margin: 4,
            cursor: 'pointer',
            boxShadow: '0 4 8 0 rgba(0,0,0,0.2)',
            transition: '0.3s',
        },
        text: {
            textAlign: 'center',
            fontWeight: 'bold',
            color: 'white',
        },
        buttonText: {
            textAlign: 'center',
            fontWeight: 'bold',
            color: 'black',
        }
    });

    return (
        <View style={styles.smallRectangle}>
            <Text style={styles.text}>La tua privacy è importante per noi. Per continuare, devi accettare le nostre politiche di privacy.</Text>
            <Button
                onPress={handleAcceptDisclaimer}
                className="p-1 bg-yellow"
                textColor="black"
                >
                <Text className="text-black font-bold">
                    Accetta
                </Text>
            </Button>
        </View>
    );
};

export default Disclaimer;
