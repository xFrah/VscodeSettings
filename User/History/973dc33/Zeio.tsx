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

    useEffect(() => {
        const { data, loading, error, refetch: userDisclaimerfetch } = useQuery(GET_DISCLAIMER_STATUS, {
            variables: { input: { profileId: profileId } },
            skip: !profileId,
        });

        const unsubscribe = navigation.addListener('state', e => {
            // Handle the state change
            console.log("fetching profile id");
            const fetchProfileID = async () => {
                const storedProfileID = await getStorageData("profileID");
                if (storedProfileID) {
                    setProfileID(storedProfileID);
                    await userDisclaimerfetch();
                }
            };

            const timeoutId = setTimeout(() => {
                fetchProfileID();
            }, 1000); 

            return () => clearTimeout(timeoutId);
        });

        if (data && data.getUserDisclaimerStatus) {
            setDisclaimerAccepted(data.getUserDisclaimerStatus.disclaimerAccepted);
        } else {
            console.log("errore");
        }
        

        return unsubscribe;
    }, [navigation, loading, data, setDisclaimerAccepted]);


    const [updateDisclaimerStatus] = useMutation(UPDATE_DISCLAIMER_STATUS);

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
