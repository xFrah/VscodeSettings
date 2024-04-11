import React from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import WorkerTimelineScreen from "./workers_navigation/WorkersTimeline";
import WorkerDashboardLayout from "./workers_navigation/";
import UserSettingPage from "../managers_view/UserSetting";
import { View, Text, TouchableOpacity } from "react-native";
// @ts-ignore
import Ionicons from "react-native-vector-icons/Ionicons";
import MaterialCommunityIcons from "react-native-vector-icons/MaterialCommunityIcons";
import WorkerDashboard from "../workers_view/worker_dashboard";

// const Tab = createBottomTabNavigator();

const Stack = createNativeStackNavigator();

// const TabDetails = {
//   WorkerDashboard : {
//     name: 'WorkerDashboard',
//     icon: 'home-outline'
//   },
//   WorkerTimeline : {
//     name: 'WorkerTimeline',
//     icon: 'calendar-outline'
//   },
//   Setting : {
//     name: 'Setting',
//     icon: 'settings-outline'
//   }
// }
// <Tab.Navigator
//   screenOptions={({ route }) => ({
//     headerShown: false,
//     tabBarStyle: {
//       backgroundColor: "#171b27",
//       height: 90,
//       borderTopColor: "#171b27",
//       position: "relative",
//     },
//     title: "",
//     tabBarIcon: ({ focused, color, size }) => {
//       const iconName = TabDetails[route.name].icon;
//       // You can return any component that you like here!
//       return <Ionicons name={iconName} size={25} color={color} />;
//     },
//     tabBarActiveTintColor: "white",
//     tabBarInactiveTintColor: "#6f7178",
//   })}
// >
//   <Tab.Screen name="WorkerDashboard" component={WorkerDashboardLayout} />
//   <Tab.Screen name="WorkerTimeline" component={WorkerTimelineScreen} />
//   <Tab.Screen name="Setting" component={UserSettingPage} />
// </Tab.Navigator>

function WorkersLayout({ navigation }) {
  const Stack = createNativeStackNavigator();
  return (
    <>
      <View className="flex-1 flex-row bg-secondary">
        <View className="flex flex-col items-start gap-2 w-2/12 top-0">
          <View className="w-full h-20 bg-yellow items-center justify-center">
            <Text
              className="text-primary font-bold text-2xl"
              style={{ fontFamily: "Lato" }}
            >
              ReportOne
            </Text>
          </View>

          <TouchableOpacity
            onPress={() => {
              navigation.navigate("Dashboard");
            }}
            className="flex flex-col w-full items-start justify-center p-4"
          >
            <View className="items-center justify-center flex flex-row gap-4">
              <Ionicons name={"pie-chart-outline"} size={35} color={"white"} />
              <Text
                className="text-white font-bold text-lg"
                style={{ fontFamily: "Lato" }}
              >
                Dashboard
              </Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={() => {
              navigation.navigate("Projects");
            }}
            className="flex flex-col w-full items-start justify-center p-4"
          >
            <View className="items-center justify-center flex flex-row gap-4">
              <MaterialCommunityIcons
                name={"clipboard-list-outline"}
                size={35}
                color={"white"}
              />
              <Text
                className="text-white font-bold text-lg"
                style={{ fontFamily: "Lato" }}
              >
                Projects
              </Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={() => {
              navigation.navigate("WorkerTimeline");
            }}
            className="flex flex-col w-full items-start justify-center p-4"
          >
            <View className="items-center justify-center flex flex-row gap-4">
              <Ionicons name={"hourglass-outline"} size={35} color={"white"} />
              <Text
                className="text-white font-bold text-lg"
                style={{ fontFamily: "Lato" }}
              >
                Timeline
              </Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={() => {
              navigation.navigate("Setting");
            }}
            className="flex flex-col w-full items-start justify-center p-4"
          >
            <View className="items-center justify-center flex flex-row gap-4">
              <Ionicons name={"settings-outline"} size={35} color={"white"} />
              <Text className="text-white font-bold text-lg">Settings</Text>
            </View>
          </TouchableOpacity>
        </View>

        <View className="flex-1 bg-primary p-10">
          <Stack.Navigator
            screenOptions={({ route }) => ({
              headerShown: false,
              title: "",
              headerBackVisible: false,
            })}
          >
            <Stack.Screen name="Dashboard" component={WorkerDashboard} />
            <Stack.Screen name="Projects" component={WorkerDashboardLayout} />
            <Stack.Screen
              name="WorkerTimeline"
              component={WorkerTimelineScreen}
            />

            <Stack.Screen name="Setting" component={UserSettingPage} />
          </Stack.Navigator>
        </View>
      </View>
    </>
  );
}

export default WorkersLayout;
