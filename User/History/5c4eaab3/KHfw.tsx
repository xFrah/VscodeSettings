import * as React from "react";
import { View, SafeAreaView } from "react-native";
import {
  Button,
  TextInput,
  HelperText,
  Text,
  ActivityIndicator,
} from "react-native-paper";
import { gql, useMutation, useQuery } from "@apollo/client";
import { Formik } from "formik";
import { getStorageData, storeData } from "../libs/storage";
import { USER_TOKEN_KEY } from "../constants/App.constants";
import * as SplashScreen from "expo-splash-screen";
import { useCallback, useEffect, useState } from "react";

SplashScreen.preventAutoHideAsync();

const LOGIN_MUTATION = gql`
  mutation login($input: loginInput) {
    login(input: $input) {
      token
    }
  }
`;

const getUserProfileQuery = gql`
  query getUserProfile {
    getUserProfile {
      role
    }
  }
`;

export default function Login({ navigation }: any) {
  const [appIsReady, setAppIsReady] = useState(false);
  const [isPasswordHidden, setisPasswordHidden] = useState(true);

  const navigateToRoleDashboards = async () => {
    const profileData = await userProfilerefetch();
    await storeData("role", profileData?.data?.getUserProfile?.role);
    const ADMIN_ROLES = ["ADMIN", "AREA_MANAGER", "PROJECT_MANAGER"];
    if (
      profileData?.data?.getUserProfile?.role &&
      ADMIN_ROLES.includes(profileData?.data?.getUserProfile?.role)
    ) {
      navigation.push("ManagersDashboard");
    } else if (
      profileData?.data?.getUserProfile?.role &&
      profileData?.data?.getUserProfile?.role === "CONSULTANT"
    ) {
      navigation.push("");
    }
  };

  const [loginUser, { data, loading, error }] = useMutation(LOGIN_MUTATION);

  const { data: userProfileData, refetch: userProfilerefetch } =
    useQuery(getUserProfileQuery);

  const checkIsTokenPresent = async () => {
    await getStorageData(USER_TOKEN_KEY)
      .then(async (tokenCurrent) => {
        setAppIsReady(true);
        if (tokenCurrent) {
          await navigateToRoleDashboards();
        }
      })
      .catch(() => {
        console.log(error);
      });
  };

  useEffect(() => {
    checkIsTokenPresent();
    if (data) {
      storeData(USER_TOKEN_KEY, data.login.token)
        .then(async () => {
          await navigateToRoleDashboards();
        })
        .catch((error) => {
          console.log(error);
        });
    }
    if (error) {
      console.log(error.graphQLErrors, "login error");
    }
  }, [data]);

  const onLayoutRootView = useCallback(async () => {
    if (appIsReady) {
      await SplashScreen.hideAsync();
    }
  }, [appIsReady]);

  if (!appIsReady) {
    return null;
  }

  async function login(values) {
    try {
      await loginUser({ variables: { input: values } });
    } catch (error) {
      console.log(error, "error in login");
    }
  }

  return (
    <View onLayout={onLayoutRootView}>
      <View className="flex flex-col bg-primary h-screen items-center justify-start">
        <Text
          className="text-white font-bold text-8xl my-24"
          style={{ fontFamily: "Lato" }}
        >
          ReportOne
        </Text>
        <View className="w-2/5 p-10 first-letter: rounded-xl border-2 border-bordercolor">
          <Text
            variant="displayMedium"
            className="text-white pb-4 font-semibold text-center"
          >
            Login
          </Text>
          <SafeAreaView>
            <Formik
              initialValues={{ email: "", password: "" }}
              onSubmit={login}
            >
              {({ handleChange, handleBlur, handleSubmit, values }) => (
                <View className="gap-10">
                  <TextInput
                    label="Email"
                    right={<TextInput.Icon icon="email" iconColor="white" />}
                    onChangeText={handleChange("email")}
                    onBlur={handleBlur("email")}
                    value={values.email}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineStyle={{
                      borderWidth: 1,
                      borderColor: "#53555e",
                    }}
                  />

                  <TextInput
                    label="Password"
                    secureTextEntry={isPasswordHidden}
                    onChangeText={handleChange("password")}
                    onBlur={handleBlur("password")}
                    value={values.password}
                    right={
                      <TextInput.Icon
                        icon="eye"
                        iconColor="white"
                        onPress={() => setisPasswordHidden(!isPasswordHidden)}
                      />
                    }
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineStyle={{
                      borderWidth: 1,
                      borderColor: "#53555e",
                    }}
                  />

                  {error?.graphQLErrors &&
                    error.graphQLErrors.map((err, idx) => {
                      return (
                        <HelperText
                          type="error"
                          visible={!!error?.graphQLErrors.length}
                          key={idx}
                        >
                          {err.message}
                        </HelperText>
                      );
                    })}

                  <View className="py-2 items-center justify-center">
                    <Button
                      mode="contained"
                      onPress={handleSubmit as (e: unknown) => void}
                      className="p-1 bg-yellow"
                    >
                      {loading ? (
                        <ActivityIndicator animating={true} color={"red"} />
                      ) : (
                        <Text className="text-black text-2xl font-bold">
                          Login
                        </Text>
                      )}
                    </Button>
                  </View>
                </View>
              )}
            </Formik>
          </SafeAreaView>
        </View>
      </View>
    </View>
  );
}
