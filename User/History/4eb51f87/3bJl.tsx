import React from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import WorkerProjectListScreen from "../../workers_view/project_list";
import WorkerProjectDetailsScreen from "../../workers_view/project_details";
import WorkerActivityDetailsScreen from "../../workers_view/activity_details";
import WorkerCreateNewTask from "../../workers_view/create_new_task";
import { APP_SLIDE_ANIMATION } from "../../../constants/App.constants";
import TaskDetails from "../../workers_view/task_details";
import { DisclaimerProvider } from '../../../components/disclaimerContext';

const Stack = createNativeStackNavigator();

function WorkerDashboardLayout() {
  return (
    <DisclaimerProvider>
      <Stack.Navigator
        screenOptions={{ headerStyle: { backgroundColor: "#2a2c38" } }}
        initialRouteName="WorkerProjectListScreen"
      >
        <Stack.Screen
          name="WorkerProjectListScreen"
          component={WorkerProjectListScreen}
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
          name="WorkerProjectDetailsScreen"
          component={WorkerProjectDetailsScreen}
          options={{
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            animation: APP_SLIDE_ANIMATION,
            headerShadowVisible: false,
            headerTitle: () => <></>,
          }}
        />

        <Stack.Screen
          name="WorkerActivityDetailsScreen"
          component={WorkerActivityDetailsScreen}
          options={{
            headerBackTitleVisible: false,
            headerTintColor: "#FFFFFF",
            headerShadowVisible: false,
            animation: APP_SLIDE_ANIMATION,
            headerTitle: () => <></>,
          }}
        />

        <Stack.Screen
          name="WorkerCreateNewTask"
          component={WorkerCreateNewTask}
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

export default WorkerDashboardLayout;
