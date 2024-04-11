import * as React from "react";
import { ScrollView, View } from "react-native";
import { Button, Text } from "react-native-paper";
import { USER_TOKEN_KEY } from "../../constants/App.constants";
import { removeStorageData, storeData } from "../../libs/storage";
import { gql, useApolloClient, useMutation, useQuery } from "@apollo/client";
import Loader from "../../components/Loader";
import { useDisclaimer } from '../../components/disclaimerContext';
import { useAuth } from '../AuthContext';


const GET_WORKER_TOKEN = gql`
  mutation switchRole($input: SwitchRoleInput) {
    switchRole(input: $input) {
      token
    }
  }
`;

const GET_USER_OTHER_ROLE = gql`
  query getUserProfile {
    getUserProfile {
      role
      last_name
      first_name
      otherProfiles {
        role {
          role
          id
        }
      }
    }
  }
`;

function UserSettingPage({ navigation }) {
  const client = useApolloClient();
  const { resetDisclaimerStatus } = useDisclaimer();
  const { setIsAuthenticated } = useAuth();

  const {
    data: userRole,
    refetch: userRoleRefetch,
    loading: loadingUserRole,
  } = useQuery(GET_USER_OTHER_ROLE);

  const [switchProfileOfUser, { data: newToken, loading, error }] =
    useMutation(GET_WORKER_TOKEN);

  async function logout() {
    await removeStorageData(USER_TOKEN_KEY);
    await removeStorageData("role");
    await removeStorageData("profileID");
    resetDisclaimerStatus(false);
    setIsAuthenticated(false);
    client.clearStore();
    navigation.push("Login");
  }

  const navigateToRoleDashboards = async () => {
    const profileData = await userRoleRefetch();
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
      navigation.push("WorkersDashboard");
    }
  };

  const switchProfile = async (profileID) => {
    if (profileID) {
      let token = await switchProfileOfUser({
        variables: { input: { role_id: profileID } },
      });

      await removeStorageData(USER_TOKEN_KEY);
      await removeStorageData("role");
      await removeStorageData("profileID");
      resetDisclaimerStatus(false);
      await client.clearStore();

      await storeData(USER_TOKEN_KEY, token?.data?.switchRole?.token)
        .then(async () => {
          await navigateToRoleDashboards();
        })
        .catch((error) => {
          console.log(error);
        });
    }
  };

  return (
    <>
      <View className="flex flex-col p-2 bg-primary h-full">
        <Loader loading={loadingUserRole} />
        <ScrollView
          contentInsetAdjustmentBehavior="automatic"
          bounces={false}
          showsHorizontalScrollIndicator={false}
        >
          <Text
            variant="displaySmall"
            className="text-white pl-2 pb-4 font-semibold"
          >
            Settings
          </Text>
          <View className="items-center h-full p-6">
            {/* <Avatar.Icon size={200} icon="account" style={{backgroundColor: "#ffdf6b"}} color="#2a2c38"/> */}
            <View className="items-center justify-center w-full my-8">
              <Text variant="displayMedium" className="text-white font-bold">
                {userRole?.getUserProfile?.first_name}
              </Text>
              <Text variant="displayMedium" className="text-white font-bold">
                {userRole?.getUserProfile?.last_name}
              </Text>
            </View>
            <View className="gap-8 items-center w-1/2 justify-center">
              <Button
                onPress={logout}
                mode={"outlined"}
                buttonColor="#ffdf6b"
                textColor="#2a2c38"
                className="w-full"
              >
                <Text className="text-black text-2xl font-bold"> Logout</Text>
              </Button>

              {userRole?.getUserProfile?.otherProfiles?.length
                ? userRole?.getUserProfile?.otherProfiles.map(
                    (element, key) => {
                      return (
                        <Button
                          key={key}
                          onPress={() => switchProfile(element.role.id)}
                          mode={"outlined"}
                          buttonColor="#ffdf6b"
                          textColor="#2a2c38"
                          className="w-full"
                        >
                          <Text className="text-black text-2xl font-bold">
                            Switch profile to{" "}
                            {element.role.role == "CONSULTANT"
                              ? "Worker"
                              : null}
                            {element.role.role == "PROJECT_MANAGER"
                              ? "Project manager"
                              : null}
                            {element.role.role == "AREA_MANAGER"
                              ? "Area manager"
                              : null}
                            {element.role.role == "ADMIN" ? "Admin" : null}
                          </Text>
                        </Button>
                      );
                    }
                  )
                : null}
            </View>
          </View>
        </ScrollView>
      </View>
    </>
  );
}

export default UserSettingPage;
