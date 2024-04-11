import React from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import Clientlist from "../../managers_view/client_list";
import ActivityList from "../../managers_view/activity_list";
import ProjectList from "../../managers_view/project_list";
import ProjectDetails from "../../managers_view/project_details";
import ActivityDetails from "../../managers_view/activity_details";
import CreateNewProject from "../../managers_view/create_new_project";
import CreateNewActivity from "../../managers_view/create_new_activity";
import { View, Text } from "react-native";
import CreateNewClient from "../../managers_view/create_new_client";
import WorkerDetails from "../../managers_view/worker_details";
import { APP_SLIDE_ANIMATION } from "../../../constants/App.constants";
import Dashboard from "../../managers_view/dashboard";
import TaskDetails from "../../managers_view/task_details";
import WorkerCreateNewTask from "../../workers_view/create_new_task";
import { DisclaimerProvider } from '../../../components/disclaimerContext';

const Stack = createNativeStackNavigator();

function ClientLayout() {
  return (
    <DisclaimerProvider>
      <Stack.Navigator
        screenOptions={{ headerStyle: { backgroundColor: "#2a2c38" } }}
        initialRouteName="ClientList"
      >
        <Stack.Screen
          name="ClientList"
          component={Clientlist}
          options={{
            headerShown: false,
            gestureEnabled: false,
            headerBackVisible: false,
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
          }}
        />

        <Stack.Screen
          name="ActivityList"
          component={ActivityList}
          options={{
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
          }}
        />

        <Stack.Screen
          name="ProjectList"
          component={ProjectList}
          options={{
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
          }}
        />

        <Stack.Screen
          name="ProjectDetails"
          component={ProjectDetails}
          options={{
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
          }}
        />

        <Stack.Screen
          name="ProjectOverview"
          component={Dashboard}
          options={{
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
          }}
        />

        <Stack.Screen
          name="ActivityDetails"
          component={ActivityDetails}
          options={{
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
          }}
        />

        <Stack.Screen
          name="CreateNewProject"
          component={CreateNewProject}
          options={{
            headerBackVisible: true,
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
          }}
        />

        <Stack.Screen
          name="CreateNewActivity"
          component={CreateNewActivity}
          options={{
            headerBackVisible: true,
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
          }}
        />

        <Stack.Screen
          name="CreateNewClient"
          component={CreateNewClient}
          options={{
            headerBackVisible: true,
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
            headerRight: () => {
              return (
                <View className="justify-end items-end pr-2">
                  <Text className="text-white">Add Client</Text>
                </View>
              );
            },
          }}
        />

        <Stack.Screen
          name="EditClientDetail"
          component={CreateNewClient}
          options={{
            headerBackVisible: true,
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
            headerRight: () => {
              return (
                <View className="justify-end items-end pr-2">
                  <Text className="text-white">Edit Client</Text>
                </View>
              );
            },
          }}
        />

        <Stack.Screen
          name="EditProjectDetails"
          component={CreateNewProject}
          options={{
            headerBackVisible: true,
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
          }}
        />

        <Stack.Screen
          name="EditActivityDetails"
          component={CreateNewActivity}
          options={{
            headerBackVisible: true,
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
          }}
        />

        <Stack.Screen
          name="WorkerDetails"
          component={WorkerDetails}
          options={{
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
          }}
        />

        <Stack.Screen
          name="TaskDetails"
          component={TaskDetails}
          options={{
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
          }}
        />

        <Stack.Screen
          name="EditTask"
          component={WorkerCreateNewTask}
          options={{
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
          }}
        />
      </Stack.Navigator>
    </DisclaimerProvider>
  );
}

export default ClientLayout;
