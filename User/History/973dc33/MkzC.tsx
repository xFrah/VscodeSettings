// DisclaimerContext.tsx
import React, { createContext, useContext, useEffect, useState } from 'react';
import { useQuery, useMutation, gql } from "@apollo/client";
import { getStorageData } from "../libs/storage";
import { useNavigation } from '@react-navigation/native';

const DisclaimerContext = createContext(null);

export const useDisclaimer = () => useContext(DisclaimerContext);

const GET_DISCLAIMER_STATUS = gql`
  query GetUserDisclaimerStatus($input: getUserDisclaimerStatusInput) {
    getUserDisclaimerStatus(input: $input) {
      disclaimerAccepted
    }
  }
`;

const UPDATE_DISCLAIMER_STATUS = gql`
  mutation UpdateUserDisclaimerStatus($input: updateUserDisclaimerStatusInput) {
    updateUserDisclaimerStatus(input: $input) {
      success
      message
    }
  }
`;

export const DisclaimerProvider = ({ children }) => {
    const [disclaimerAccepted, setDisclaimerAccepted] = useState(false);
    const [profileId, setProfileID] = useState(null);
    const navigation = useNavigation();

    const { data, loading, error } = useQuery(GET_DISCLAIMER_STATUS, {
        variables: { input: { profileId: profileId } },
        skip: !profileId,
    });

    useEffect(() => {
        const unsubscribe = navigation.addListener('state', e => {
            const fetchProfileID = async () => {
                const storedProfileID = await getStorageData("profileID");
                if (storedProfileID) {
                    setProfileID(storedProfileID);
                }
            };
            console.log(data, "sdkjdfhjgfgj");

            const timeoutId = setTimeout(() => {
                fetchProfileID();
            }, 1000); 

            return () => clearTimeout(timeoutId); 
        });

        return unsubscribe;
    }, [navigation]);

    console.log(profileId, "id");

    console.log(data, "data");

    const [updateDisclaimerStatus] = useMutation(UPDATE_DISCLAIMER_STATUS);

    useEffect(() => {
        const prova = navigation.addListener('state', e => {
            if (data && data.getUserDisclaimerStatus) {
                console.log(data.getUserDisclaimerStatus.disclaimerAccepted, "disclaimerAccepted");
                setDisclaimerAccepted(data.getUserDisclaimerStatus.disclaimerAccepted);
                console.log(data.getUserDisclaimerStatus.disclaimerAccepted, " cazz");
            } else {
                console.log("errore");
            }
        });

        return prova;
    }, [navigation, loading, data, setDisclaimerAccepted]);

    const handleAcceptDisclaimer = async () => {
        try {
            const response = await updateDisclaimerStatus({
                variables: { input: { profileId: profileId, accepted: true } }
            });
            if (response.data.updateUserDisclaimerStatus.success) {
                setDisclaimerAccepted(true);
            }
        } catch (error) {
            console.log("altro errore")
        }
    };

    const resetDisclaimerStatus = () => {
        setDisclaimerAccepted(false);
    };

    return (
        <DisclaimerContext.Provider value={{ disclaimerAccepted, handleAcceptDisclaimer, resetDisclaimerStatus, }}>
            {children}
        </DisclaimerContext.Provider>
    );
};
