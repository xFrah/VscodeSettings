import {
  ApolloClient,
  InMemoryCache,
  ApolloProvider,
  createHttpLink,
} from "@apollo/client";
import { Provider, configureFonts, DefaultTheme } from "react-native-paper";
import { NativeWindStyleSheet } from "nativewind";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { setContext } from "@apollo/client/link/context";
import Login from "./src/views/login";
import { en, registerTranslation } from "react-native-paper-dates";
import { getStorageData } from "./src/libs/storage";
import { USER_TOKEN_KEY } from "./src/constants/App.constants";
import ManagersLayout from "./src/views/layouts/ManagersLayout";
import WorkersLayout from "./src/views/layouts/WorkersLayout";
import { DisclaimerProvider } from './src/components/disclaimerContext';
import Disclaimer from './src/components/disclaimer';
import { AuthProvider } from './src/views/AuthContext';
import React, { useEffect, useState } from "react";
import * as Font from "expo-font";
import "./App.css";
import "./src/locales/index"
import { useTranslation } from "react-i18next";
import { View } from "react-native";
registerTranslation("en", en);

const authLink = setContext(async (_, { headers }) => {
  // get the authentication token from local storage if it exists
  const token = await getStorageData(USER_TOKEN_KEY);
  // return the headers to the context so httpLink can read them
  return {
    headers: {
      ...headers,
      authorization: token ? `Bearer ${token}` : "",
    },
  };
});

// PROD URL: http://api.reportone.it/
// tunnel url : http://trackapi.loclx.io
// local IP : http://192.168.1.9:4000/

const httpLink = createHttpLink({
  uri: "http://localhost:5432/",
});

const cache = new InMemoryCache({
  addTypename: false,
  typePolicies: {
    Query: {
      fields: {
        getUserProfile: {
          merge(existing = {}, incoming) {
            // Custom merge logic
            return { ...existing, ...incoming, role: incoming.role };
          },
        },
      },
    },
  },
});

const client = new ApolloClient({
  link: authLink.concat(httpLink),
  cache: cache,
});

const Stack = createNativeStackNavigator();

NativeWindStyleSheet.setOutput({
  default: "native",
});

function animationChange() {
  console.log("animated");
}
const config = {
  screens: {
    ManagersDashboard: "admin",
  },
};

const linking = {
  prefixes: [],
  config,
};

export default function App() {
  const [isFontLoaded, setIsFontLoaded] = useState(false);

  const customFonts = {
    Lato: require("./Public/font/Lato-Regular.ttf"),
    Montserrat: require("./Public/font/Montserrat-Regular.ttf"),
    Verdana: require("./Public/font/verdana.ttf"),
  };

  const theme = {
    ...DefaultTheme,
    fonts: configureFonts({
      config: {
        fontFamily: "Lato",
      },
    }),
  };

  const _loadFontsAsync = async () => {
    await Font.loadAsync(customFonts);
    setIsFontLoaded(true);
  };

  useEffect(() => {
    if (!isFontLoaded) {
      _loadFontsAsync;
    }
  }, [isFontLoaded]);

  const { i18n, t } = useTranslation();
  const setLanguage = async () => {
    await i18n
      .changeLanguage("it")
      .then(() => console.log("language set to italian"))
      .catch((err) => console.log(err));
  };

  return (
    <Provider theme={theme}>
      <ApolloProvider client={client}>
        <AuthProvider>
          <DisclaimerProvider>
            <View onLayout={setLanguage} className="flex-1">
              <NavigationContainer linking={linking}>
                <Stack.Navigator
                  screenOptions={{
                    headerStyle: { backgroundColor: "#2a2c38" },
                    animation: "none",
                  }}
                  initialRouteName="Login"
                  screenListeners={{ transitionStart: animationChange }}
                >
                  <Stack.Screen
                    name="Login"
                    component={Login}
                    options={{ headerShown: false }}
                  />
                  <Stack.Screen
                    name="ManagersDashboard"
                    component={ManagersLayout}
                    options={{
                      headerShown: false,
                      gestureEnabled: false,
                      animation: "none",
                      animationTypeForReplace: "push",
                    }}
                  />
                  <Stack.Screen
                    name="WorkersDashboard"
                    component={WorkersLayout}
                    options={{ headerShown: false, gestureEnabled: false }}
                  />
                </Stack.Navigator>
                <Disclaimer />
              </NavigationContainer>
            </View>
          </DisclaimerProvider>
        </AuthProvider>
      </ApolloProvider>
    </Provider>
  );
}