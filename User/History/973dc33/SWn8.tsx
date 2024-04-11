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
    let [disclaimerAccepted, setDisclaimerAccepted] = useState(false);
    const [profileId, setProfileID] = useState(null);
    const navigation = useNavigation();

    useEffect(() => {
        const unsubscribe = navigation.addListener('state', e => {
            const fetchProfileID = async () => {
                const storedProfileID = await getStorageData("profileID");
                if (storedProfileID) {
                    setProfileID(storedProfileID);
                }
            };

            fetchProfileID();
        });

        return unsubscribe;
    }, [navigation]);

    console.log(profileId, "id");

    const { data, loading, error } = useQuery(GET_DISCLAIMER_STATUS, {
        variables: { input: { profileId: profileId } },
        skip: !profileId,
    });

    useEffect(() => {
        if (data && data.getUserDisclaimerStatus) {
            setDisclaimerAccepted(data.getUserDisclaimerStatus.disclaimerAccepted);
        }
    }, [data]);
    
    const [updateDisclaimerStatus] = useMutation(UPDATE_DISCLAIMER_STATUS);

    useEffect(() => {
        const prova = navigation.addListener('state', e => {
            if (data && data.getUserDisclaimerStatus) {
                console.log(data.getUserDisclaimerStatus.disclaimerAccepted, "disclaimerAccepted");
                setDisclaimerAccepted(data.getUserDisclaimerStatus.disclaimerAccepted);
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
                console.log(response.data.updateUserDisclaimerStatus.success, "risposta");
                setDisclaimerAccepted(true);
                console.log(disclaimerAccepted, "dentro");
            }
        } catch (error) {
            console.log("altro errore")
        }
    };

    console.log(disclaimerAccepted, "fuori");
    const resetDisclaimerStatus = () => {
        setDisclaimerAccepted(false);
    };

    return (
        <DisclaimerContext.Provider value={{ disclaimerAccepted, handleAcceptDisclaimer, resetDisclaimerStatus }}>
            {children}
        </DisclaimerContext.Provider>
    );
};
