import { gql, useMutation, useQuery } from "@apollo/client";
import { Formik } from "formik";
import React, { useEffect, useState } from "react";
import { TouchableOpacity, View } from "react-native";
import { ScrollView } from "react-native-gesture-handler";
import {
  Button,
  TextInput,
  Modal,
  Portal,
  Text,
  ActivityIndicator,
  Checkbox,
} from "react-native-paper";
import DropDown from "react-native-paper-dropdown";
import MyWebDatePicker from "../../components/MyDatePicker";
import SearchableDropdown from "../../components/SearchableDropdown";
import { validate } from "../../libs/form.validate";
import { convertToFloat } from "../../libs/shared.services";
import { useTranslation } from "react-i18next";

const createProjectQuery = gql`
  mutation createProject($input: createProjectInput) {
    createProject(input: $input) {
      status
      message
    }
  }
`;
const getAllAreaManager = gql`
  query findUserByRole($input: findUserByRoleInput) {
    findUserByRole(input: $input) {
      id
      user {
        first_name
        last_name
      }
    }
  }
`;
const getAllProjectManager = gql`
  query findUserByRole($input: findUserByRoleInput) {
    findUserByRole(input: $input) {
      id
      user {
        first_name
        last_name
      }
    }
  }
`;
const getAllArea = gql`
  query getAllAreas {
    getAllAreas {
      id
      name
    }
  }
`;
const getClientDetails = gql`
  query getClientByID($input: getAllProjectInput) {
    getClientByID(input: $input) {
      address
      id
      name
    }
  }
`;
const getProjectDetailsQuery = gql`
  query getProjectById($input: userProfileInput) {
    getProjectById(input: $input) {
      address
      client_id
      area_manager {
        id
        user {
          first_name
          last_name
        }
      }
      billing_time
      budget
      client_note
      description
      end_date
      event_date
      id
      invoice_id
      name
      note
      offer_id
      project_area {
        id
        name
      }
      project_created_by {
        user {
          first_name
          last_name
        }
      }
      project_id
      project_manager {
        id
        user {
          first_name
          last_name
        }
      }
      start_date
      state_code
      year
      project_reference
      client_reference
      status_code
      contract_value
      margin
      fee
      payment_check
      reporting_file
      check_planning
      quality_plan
      expeses_to_be_reimbursed
      expeses_included
      status
      client_group
      client_manager
      invoice_details
      billing_time
      check
      project_sector {
        id
      }
      note_compilatore
      note_check_pm
      note_resp_area
    }
  }
`;
const EDIT_PROJECT_DETAILS = gql`
  mutation updateProject($input: updateProjectInput) {
    updateProject(input: $input) {
      status
      message
    }
  }
`;
const GET_ALL_SECTORS = gql`
  query getAllSectors {
    getAllSectors {
      id
      name
    }
  }
`;

function CreateNewProject({ navigation, route }) {
  const [clientid] = useState(route.params.id);
  const [projectid] = useState(route?.params?.projectID);
  const [validationErrors, setValidationErrors] = useState({});
  const [coordinates, setCoordinates] = useState({
    latitude: "",
    longitude: "",
  });
  const { t } = useTranslation();

  const [showProjectStatusDropdown, setShowProjectStatusDropdown] =
    useState(false);

  const REQUIRED_FIELDS = [
    "name",
    "client_manager",
    "client_group",
    "project_id",
    "area_id",
    "sector_id",
    "project_manager_id",
    "project_reference",
    "contract_value",
    "fee",
    "margin",
    "budget",
  ];

  const PROJECT_STATUS = [
    { label: "Da Confermare", value: "Da_confermare" },
    { label: "In Programmazione", value: "In_Programmazione" },
    { label: "Da Fatturare", value: "Da_Fatturare" },
    { label: "Aperta", value: "Aperta" },
    { label: "Annuale", value: "Annuale" },
    { label: "Chiusa", value: "Chiusa" },
    { label: "Annullata", value: "Annullata" },
  ];

  const [showSectorDropDown, setShowSectorDropDown] = useState(false);

  const { data: sectorList, refetch: refetchSectorList } =
    useQuery(GET_ALL_SECTORS);

  let formattedSectorList = [];

  if (sectorList?.getAllSectors?.length) {
    formattedSectorList = sectorList?.getAllSectors?.map((sector) => {
      return {
        label: sector.name,
        value: sector.id,
      };
    });
  }

  const { data: projectData, refetch: refetchProjectData } = useQuery(
    getProjectDetailsQuery,
    {
      variables: { input: { id: projectid } },
    }
  );


  const { data: clientData, refetch: refetchClient } = useQuery(
    getClientDetails,
    {
      variables: {
        input: {
          client_id: clientid || projectData?.getProjectById?.client_id,
        },
      },
    }
  );

  const [createProject, { data, loading, error }] =
    useMutation(createProjectQuery);

  // area manager dropdown logic
  const [showAMDropDown, setshowAMDropDown] = useState(false);

  const {
    loading: areaManagerloading,
    error: areaManagererror,
    data: areaManagerdata,
  } = useQuery(getAllAreaManager, {
    variables: { input: { role: "AREA_MANAGER" } },
  });

  let formattedAreaManagers = [];

  if (areaManagerdata?.findUserByRole?.length) {
    formattedAreaManagers = areaManagerdata?.findUserByRole.map((manager) => {
      return {
        label: `${manager.user?.first_name} ${manager?.user?.last_name}`,
        value: manager?.id,
      };
    });
  }
  // ends here

  // Project manager dropdown logic
  const [showPMDropDown, setshowPMDropDown] = useState(false);

  const {
    loading: projectManagerloading,
    error: projectManagerrerror,
    data: projectManagerdata,
  } = useQuery(getAllProjectManager, {
    variables: { input: { role: "PROJECT_MANAGER" } },
  });

  let formattedProjectManagers = [];

  if (projectManagerdata?.findUserByRole?.length) {
    formattedProjectManagers = projectManagerdata?.findUserByRole.map(
      (manager) => {
        return {
          label: `${manager.user?.first_name} ${manager?.user?.last_name}`,
          value: manager?.id,
        };
      }
    );
  }
  // ends here

  // all areas dropdown logic
  const [showAreasDropDown, setshowAreasDropDown] = useState(false);

  const {
    loading: areasloading,
    error: areaserrerror,
    data: areasdata,
  } = useQuery(getAllArea);

  let formattedAreas = [];

  if (areasdata?.getAllAreas?.length) {
    formattedAreas = areasdata?.getAllAreas.map((area) => {
      return {
        label: area?.name,
        value: area?.id,
      };
    });
  }
  // ends here

  const [isEventDatePickerVisible, setisEventDatePickerVisible] =
    useState(false);
  const [isStartDatePickerVisible, setisStartDatePickerVisible] =
    useState(false);
  const [isEndDatePickerVisible, setisEndDatePickerVisible] = useState(false);

  const containerStyle = {
    backgroundColor: "white",
    padding: 20,
    alignSelf: "center",
  };

  const submitData = async (values) => {
    convertToFloat(values, "budget");
    convertToFloat(values, "fee");
    convertToFloat(values, "margin");
    convertToFloat(values, "contract_value");

    setValidationErrors({});

    let isValidForm = await validate(
      values,
      setValidationErrors,
      REQUIRED_FIELDS
    );

    if (values?.margin < 0 || values?.margin > 100) {
      isValidForm = false;
      return setValidationErrors({
        margin: "Fixed Cost percentage cannot be greater than 100",
      });
    }

    if (values?.fee < 0 || values?.fee > 100) {
      isValidForm = false;
      return setValidationErrors({
        fee: "Fee percentage cannot be greater than 100",
      });
    }

    try {
      if (isValidForm) {
        if (projectid) {
          await editProject({
            variables: { input: { ...values, id: projectid } },
          });
          navigation.navigate("ProjectList", {
            id: projectData?.getProjectById?.client_id,
          });
        } else {
          await createProject({
            variables: {
              input: {
                ...values,
                latitude: coordinates?.latitude
                  ? String(coordinates?.latitude)
                  : null,
                longitude: coordinates?.latitude
                  ? String(coordinates?.latitude)
                  : null,
              },
            },
          });
          navigation.navigate("ProjectList", {
            id: clientid,
          });
        }
      }
    } catch (error) {
      setValidationErrors({ backendError: `${error?.message}*` });
    }
  };

  useEffect(() => {
    if (clientData?.getClientByID) {
      navigation.setOptions({
        headerRight: () => {
          return (
            <View className="justify-end items-end pr-2">
              <Text className="text-white">
                {clientData?.getClientByID?.name}
              </Text>
              <Text className="text-white">
                {" "}
                {clientData &&
                  clientData.getClientByID &&
                  clientData.getClientByID.address}
              </Text>
            </View>
          );
        },
      });
    }
  }, [clientData]);

  const [initialValues, setInitialValues] = useState({
    name: "",
    client_id: clientid,
    project_id: "",
    address: "",
    description: "",
    area_id: "",
    state_code: "",
    offer_id: "",
    invoice_id: "",
    area_manager_id: "",
    project_manager_id: "",
    client_note: "",
    note: "",
    project_reference: "",
    client_reference: "",
    status_code: "",
    contract_value: 0,
    margin: 0,
    fee: 0,
    budget: 0,
    payment_check: false,
    reporting_file: false,
    check_planning: false,
    quality_plan: false,
    expeses_to_be_reimbursed: false,
    expeses_included: false,
    status: "In_Programmazione",
    client_manager: "",
    client_group: "",
    sector_id: "",
    event_date: new Date(),
    start_date: new Date(),
    end_date: new Date(),
    invoice_details: "",
    check: "",
    billing_time: "",
    note_compilatore: "",
    note_check_pm: "",
    note_resp_area: "",
  });

  useEffect(() => {
    if (projectData) {
      setInitialValues({
        name: projectData?.getProjectById?.name || "",
        client_id: projectData?.getProjectById?.client_id || "",
        project_id: projectData?.getProjectById?.project_id || "",
        address: projectData?.getProjectById?.address || "",
        description: projectData?.getProjectById?.description || "",
        area_id: projectData?.getProjectById?.project_area?.id || "",
        state_code: projectData?.getProjectById?.state_code || "",
        offer_id: projectData?.getProjectById?.offer_id || "",
        invoice_id: projectData?.getProjectById?.invoice_id || "",
        area_manager_id: projectData?.getProjectById?.area_manager?.id || "",
        project_manager_id:
          projectData?.getProjectById?.project_manager?.id || "",
        client_note: projectData?.getProjectById?.client_note || "",
        note: projectData?.getProjectById?.note || "",
        event_date:
          (projectData?.getProjectById?.event_date &&
            new Date(parseInt(projectData?.getProjectById?.event_date))) ||
          new Date(),
        start_date:
          (projectData?.getProjectById?.start_date &&
            new Date(parseInt(projectData?.getProjectById?.start_date))) ||
          new Date(),
        end_date:
          (projectData?.getProjectById?.end_date &&
            new Date(parseInt(projectData?.getProjectById?.end_date))) ||
          new Date(),
        project_reference: projectData?.getProjectById?.project_reference || "",
        client_reference: projectData?.getProjectById?.client_reference || "",
        status_code: projectData?.getProjectById?.status_code || "",
        contract_value:
          parseFloat(projectData?.getProjectById?.contract_value) || 0,
        margin: parseFloat(projectData?.getProjectById?.margin) || 0,
        fee: parseFloat(projectData?.getProjectById?.fee) || 0,
        budget: parseFloat(projectData?.getProjectById?.budget) || 0,
        payment_check: projectData?.getProjectById?.payment_check || false,
        reporting_file: projectData?.getProjectById?.reporting_file || false,
        check_planning: projectData?.getProjectById?.check_planning || false,
        quality_plan: projectData?.getProjectById?.quality_plan || false,
        expeses_to_be_reimbursed:
          projectData?.getProjectById?.expeses_to_be_reimbursed || false,
        expeses_included:
          projectData?.getProjectById?.expeses_included || false,
        status: projectData?.getProjectById?.status || "ACTIVE",
        client_manager: projectData?.getProjectById?.client_manager || "",
        client_group: projectData?.getProjectById?.client_group || "",
        sector_id: projectData?.getProjectById?.project_sector?.id || "",
        invoice_details: projectData?.getProjectById?.invoice_details || "",
        check: projectData?.getProjectById?.check || "",
        billing_time: projectData?.getProjectById?.billing_time || "",
        note_compilatore: projectData?.getProjectById?.note_compilatore || "",
        note_check_pm: projectData?.getProjectById?.note_check_pm || "",
        note_resp_area: projectData?.getProjectById?.note_resp_area || "",
      });
    }
  }, [projectData, projectid]);

  return (
    <ScrollView
      contentInsetAdjustmentBehavior="automatic"
      bounces={false}
      showsHorizontalScrollIndicator={false}
    >
      <View>
        <View className="flex flex-col p-2 pt-8 bg-primary h-full pb-24">
          <Text
            variant="displayMedium"
            className="text-white pl-2 pb-4 font-semibold"
          >
            {projectid ? "Edit Project" : "Add project"}
          </Text>

          {validationErrors["backendError"] && (
            <View>
              <Text
                className="text-red-400 text-sm p-2"
                style={{ color: "red" }}
              >
                {validationErrors["backendError"]}
              </Text>
            </View>
          )}

          <Formik
            initialValues={initialValues}
            onSubmit={submitData}
            enableReinitialize
          >
            {({
              handleChange,
              handleBlur,
              handleSubmit,
              values,
              setFieldValue,
            }) => (
              <View className="flex flex-col w-full p-2 gap-4">
                <View>
                  <TextInput
                    label={t("Name")}
                    onChangeText={handleChange("name")}
                    onBlur={handleBlur("name")}
                    value={values.name}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                  />
                  {validationErrors["name"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["name"]}
                    </Text>
                  )}
                </View>

                <View>
                  <TextInput
                    label={t("Project ID")}
                    onChangeText={handleChange("project_id")}
                    onBlur={handleBlur("project_id")}
                    value={values.project_id}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                  />
                  {validationErrors["project_id"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["project_id"]}
                    </Text>
                  )}
                </View>

                <View>
                  <TextInput
                    label={t("Address")}
                    multiline={true}
                    onChangeText={handleChange("address")}
                    onBlur={handleBlur("address")}
                    value={values.address}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                  />
                  {validationErrors["address"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["address"]}
                    </Text>
                  )}
                </View>

                {!projectid ? (
                  <View className="mt-1">
                    <SearchableDropdown setCoordinates={setCoordinates} />
                  </View>
                ) : null}

                <TextInput
                  label={t("Description")}
                  multiline={true}
                  onChangeText={handleChange("description")}
                  onBlur={handleBlur("description")}
                  value={values.description}
                  theme={{
                    colors: {
                      onSurfaceVariant: "white",
                    },
                  }}
                  className="bg-primary"
                  activeUnderlineColor="white"
                  underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                />

                <View>
                  <TextInput
                    label={t("Invoice Details")}
                    onChangeText={handleChange("invoice_details")}
                    onBlur={handleBlur("invoice_details")}
                    value={values.invoice_details}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                  />
                </View>

                <View>
                  <TextInput
                    label={t("Check")}
                    onChangeText={handleChange("check")}
                    onBlur={handleBlur("check")}
                    value={values.check}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                  />
                </View>

                <View>
                  <TextInput
                    label={t("Billing Time")}
                    onChangeText={handleChange("billing_time")}
                    onBlur={handleBlur("billing_time")}
                    value={values.billing_time}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                  />
                </View>

                <View>
                  <DropDown
                    label={t("Select Area")}
                    mode={"flat"}
                    visible={showAreasDropDown}
                    showDropDown={() => setshowAreasDropDown(true)}
                    onDismiss={() => setshowAreasDropDown(false)}
                    value={values.area_id}
                    setValue={(data) => {
                      setFieldValue("area_id", data);
                    }}
                    inputProps={{
                      activeUnderlineColor: "#53555e",
                      backgroundColor: "#2a2c38",
                      underlineColor: "#53555e",
                      borderColor: "#53555e",
                      borderBottomWidth: 2,
                      style: { backgroundColor: "#2a2c38" },
                      underlineStyle: {
                        borderWidth: 1,
                        borderColor: "#53555e",
                      },
                      right: (
                        <TextInput.Icon
                          icon={showAreasDropDown ? "menu-up" : "menu-down"}
                          iconColor="white"
                        />
                      ),
                    }}
                    list={formattedAreas}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                  />
                  {validationErrors["area_id"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["area_id"]}
                    </Text>
                  )}
                </View>

                <View>
                  <DropDown
                    label={t("Select Sector")}
                    mode={"flat"}
                    visible={showSectorDropDown}
                    showDropDown={() => setShowSectorDropDown(true)}
                    onDismiss={() => setShowSectorDropDown(false)}
                    value={values.sector_id}
                    setValue={(data) => {
                      setFieldValue("sector_id", data);
                    }}
                    inputProps={{
                      activeUnderlineColor: "#53555e",
                      backgroundColor: "#2a2c38",
                      underlineColor: "#53555e",
                      borderColor: "#53555e",
                      style: { backgroundColor: "#2a2c38" },
                      underlineStyle: {
                        borderWidth: 2,
                        borderColor: "#53555e",
                      },
                      right: (
                        <TextInput.Icon
                          icon={showSectorDropDown ? "menu-up" : "menu-down"}
                          iconColor="white"
                        />
                      ),
                    }}
                    list={formattedSectorList}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                  />
                  {validationErrors["sector_id"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["sector_id"]}
                    </Text>
                  )}
                </View>

                <View>
                  <TextInput
                    label={t("State Code")}
                    onChangeText={handleChange("state_code")}
                    onBlur={handleBlur("state_code")}
                    value={values.state_code}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                  />
                  {validationErrors["state_code"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["state_code"]}
                    </Text>
                  )}
                </View>

                <View>
                  <TextInput
                    label={t("Offer ID")}
                    onChangeText={handleChange("offer_id")}
                    onBlur={handleBlur("offer_id")}
                    value={values.offer_id}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                  />
                  {validationErrors["offer_id"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["offer_id"]}
                    </Text>
                  )}
                </View>

                <View>
                  <TextInput
                    label={t("Invoice ID")}
                    onChangeText={handleChange("invoice_id")}
                    onBlur={handleBlur("invoice_id")}
                    value={values.invoice_id}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                  />
                  {validationErrors["invoice_id"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["invoice_id"]}
                    </Text>
                  )}
                </View>

                <View>
                  <TextInput
                    label={t("Project Reference")}
                    onChangeText={handleChange("project_reference")}
                    onBlur={handleBlur("project_reference")}
                    value={values.project_reference}
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
                  {validationErrors["project_reference"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["project_reference"]}
                    </Text>
                  )}
                </View>

                <View>
                  <TextInput
                    label={t("Client Reference")}
                    onChangeText={handleChange("client_reference")}
                    onBlur={handleBlur("client_reference")}
                    value={values.client_reference}
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
                </View>

                <View>
                  <TextInput
                    label={t("Client Manager")}
                    onChangeText={handleChange("client_manager")}
                    onBlur={handleBlur("client_manager")}
                    value={values.client_manager}
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
                  {validationErrors["client_manager"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["client_manager"]}
                    </Text>
                  )}
                </View>

                <View>
                  <TextInput
                    label={t("Client Group")}
                    onChangeText={handleChange("client_group")}
                    onBlur={handleBlur("client_group")}
                    value={values.client_group}
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
                  {validationErrors["client_group"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["client_group"]}
                    </Text>
                  )}
                </View>

                <View>
                  <TextInput
                    label={t("Status Code")}
                    onChangeText={handleChange("status_code")}
                    onBlur={handleBlur("status_code")}
                    value={values.status_code}
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
                </View>

                <View>
                  <TextInput
                    label={t("Contract Value")}
                    onChangeText={handleChange("contract_value")}
                    onBlur={handleBlur("contract_value")}
                    value={values.contract_value.toString()}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineColor="white"
                    underlineStyle={{
                      borderWidth: 1,
                      borderColor: "#53555e",
                    }}
                    keyboardType="numeric"
                  />
                  {validationErrors["contract_value"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["contract_value"]}
                    </Text>
                  )}
                </View>

                <View>
                  <TextInput
                    label={t("Budget")}
                    onChangeText={handleChange("budget")}
                    onBlur={handleBlur("budget")}
                    value={values.budget.toString()}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineColor="white"
                    underlineStyle={{
                      borderWidth: 1,
                      borderColor: "#53555e",
                    }}
                    keyboardType="numeric"
                  />
                  {validationErrors["budget"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["budget"]}
                    </Text>
                  )}
                </View>

                <View>
                  <TextInput
                    label={t("Fixed Cost ( in % )")}
                    onChangeText={handleChange("margin")}
                    onBlur={handleBlur("margin")}
                    value={values.margin.toString()}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineColor="white"
                    underlineStyle={{
                      borderWidth: 1,
                      borderColor: "#53555e",
                    }}
                    keyboardType="numeric"
                  />
                  {validationErrors["margin"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["margin"]}
                    </Text>
                  )}
                </View>

                <View>
                  <TextInput
                    label="Fee ( in % )"
                    onChangeText={handleChange("fee")}
                    onBlur={handleBlur("fee")}
                    value={values.fee.toString()}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineColor="white"
                    underlineStyle={{
                      borderWidth: 1,
                      borderColor: "#53555e",
                    }}
                    keyboardType="numeric"
                  />
                  {validationErrors["fee"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["fee"]}
                    </Text>
                  )}
                </View>

                <View>
                  <DropDown
                    label={t("Select Area Manager")}
                    mode={"flat"}
                    visible={showAMDropDown}
                    showDropDown={() => setshowAMDropDown(true)}
                    onDismiss={() => setshowAMDropDown(false)}
                    value={values.area_manager_id}
                    inputProps={{
                      activeUnderlineColor: "#53555e",
                      backgroundColor: "#2a2c38",
                      underlineColor: "#53555e",
                      iconColor: "#2a2c38",
                      borderColor: "#53555e",
                      borderBottomWidth: 2,
                      style: { backgroundColor: "#2a2c38" },
                      underlineStyle: {
                        borderWidth: 1,
                        borderColor: "#53555e",
                      },
                      right: (
                        <TextInput.Icon
                          icon={showAMDropDown ? "menu-up" : "menu-down"}
                          iconColor="white"
                        />
                      ),
                    }}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    setValue={(data) => {
                      setFieldValue("area_manager_id", data);
                    }}
                    list={formattedAreaManagers}
                  />
                  {validationErrors["area_manager_id"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["area_manager_id"]}
                    </Text>
                  )}
                </View>

                <View>
                  <DropDown
                    label={t("Select Project Manager")}
                    mode={"flat"}
                    visible={showPMDropDown}
                    showDropDown={() => setshowPMDropDown(true)}
                    onDismiss={() => setshowPMDropDown(false)}
                    value={values.project_manager_id}
                    inputProps={{
                      activeUnderlineColor: "#53555e",
                      backgroundColor: "#2a2c38",
                      underlineColor: "#53555e",
                      borderBottomWidth: 2,
                      style: { backgroundColor: "#2a2c38" },
                      underlineStyle: {
                        borderWidth: 1,
                        borderColor: "#53555e",
                      },
                      right: (
                        <TextInput.Icon
                          icon={showPMDropDown ? "menu-up" : "menu-down"}
                          iconColor="white"
                        />
                      ),
                    }}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    setValue={(data) => {
                      setFieldValue("project_manager_id", data);
                    }}
                    list={formattedProjectManagers}
                  />
                  {validationErrors["project_manager_id"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["project_manager_id"]}
                    </Text>
                  )}
                </View>

                <View>
                  <TextInput
                    label={t("Client Note")}
                    onChangeText={handleChange("client_note")}
                    onBlur={handleBlur("client_note")}
                    value={values.client_note}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                  />
                  {validationErrors["client_note"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["client_note"]}
                    </Text>
                  )}
                </View>

                <TextInput
                  label={t("Note")}
                  onChangeText={handleChange("note")}
                  onBlur={handleBlur("note")}
                  value={values.note}
                  theme={{
                    colors: {
                      onSurfaceVariant: "white",
                    },
                  }}
                  className="bg-primary"
                  activeUnderlineColor="white"
                  underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                />

                <TextInput
                  label={t("Note Compilatore")}
                  onChangeText={handleChange("note_compilatore")}
                  onBlur={handleBlur("note_compilatore")}
                  value={values.note_compilatore}
                  theme={{
                    colors: {
                      onSurfaceVariant: "white",
                    },
                  }}
                  className="bg-primary"
                  activeUnderlineColor="white"
                  underlineColor="white"
                  underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                />

                <TextInput
                  label={t("Note Check PM")}
                  onChangeText={handleChange("note_check_pm")}
                  onBlur={handleBlur("note_check_pm")}
                  value={values.note_check_pm}
                  theme={{
                    colors: {
                      onSurfaceVariant: "white",
                    },
                  }}
                  className="bg-primary"
                  activeUnderlineColor="white"
                  underlineColor="white"
                  underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                />

                <TextInput
                  label="Note Resp Area"
                  onChangeText={handleChange("note_resp_area")}
                  onBlur={handleBlur("note_resp_area")}
                  value={values.note_resp_area}
                  theme={{
                    colors: {
                      onSurfaceVariant: "white",
                    },
                  }}
                  className="bg-primary"
                  activeUnderlineColor="white"
                  underlineColor="white"
                  underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                />

                {/* Event date picker with modal starts */}
                <TextInput
                  label={t("Event Date")}
                  showSoftInputOnFocus={false}
                  value={values.event_date.toDateString()}
                  right={<TextInput.Icon icon="calendar" iconColor="white" />}
                  onFocus={() =>
                    setisEventDatePickerVisible(!isEventDatePickerVisible)
                  }
                  theme={{
                    colors: {
                      onSurfaceVariant: "white",
                    },
                  }}
                  className="bg-primary"
                  activeUnderlineColor="white"
                  underlineColor="white"
                  underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                />

                <Portal>
                  <Modal
                    visible={isEventDatePickerVisible}
                    onDismiss={() => setisEventDatePickerVisible(false)}
                    // @ts-ignore
                    contentContainerStyle={containerStyle}
                  >
                    {isEventDatePickerVisible && (
                      <MyWebDatePicker
                        fieldName={"event_date"}
                        setFieldValue={setFieldValue}
                      />
                    )}
                  </Modal>
                </Portal>
                {/* Event date picker with modal Ends */}

                {/* Start date picker with modal starts */}
                <View>
                  <TextInput
                    label={t("Start Date")}
                    showSoftInputOnFocus={false}
                    value={values.start_date.toDateString()}
                    right={<TextInput.Icon icon="calendar" iconColor="white" />}
                    onFocus={() =>
                      setisStartDatePickerVisible(!isStartDatePickerVisible)
                    }
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                    className="bg-primary"
                    activeUnderlineColor="white"
                    underlineColor="white"
                    underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                  />

                  <Portal>
                    <Modal
                      visible={isStartDatePickerVisible}
                      onDismiss={() => setisStartDatePickerVisible(false)}
                      // @ts-ignore
                      contentContainerStyle={containerStyle}
                    >
                      {isStartDatePickerVisible && (
                        <MyWebDatePicker
                          fieldName={"start_date"}
                          setFieldValue={setFieldValue}
                        />
                      )}
                    </Modal>
                  </Portal>
                  {validationErrors["start_date"] && (
                    <Text className="text-red-400 p-1" style={{ color: "red" }}>
                      {validationErrors["start_date"]}
                    </Text>
                  )}
                </View>

                {/* Start date picker with modal Ends */}

                {/* End date picker with modal starts */}
                <TextInput
                  label={t("End Date")}
                  showSoftInputOnFocus={false}
                  value={values.end_date.toDateString()}
                  right={<TextInput.Icon icon="calendar" iconColor="white" />}
                  onFocus={() =>
                    setisEndDatePickerVisible(!isEndDatePickerVisible)
                  }
                  theme={{
                    colors: {
                      onSurfaceVariant: "white",
                    },
                  }}
                  className="bg-primary"
                  activeUnderlineColor="white"
                  underlineColor="white"
                  underlineStyle={{ borderWidth: 1, borderColor: "#53555e" }}
                />

                <Portal>
                  <Modal
                    visible={isEndDatePickerVisible}
                    onDismiss={() => setisEndDatePickerVisible(false)}
                    // @ts-ignore
                    contentContainerStyle={containerStyle}
                  >
                    {isEndDatePickerVisible && (
                      <MyWebDatePicker
                        fieldName={"end_date"}
                        setFieldValue={setFieldValue}
                      />
                    )}
                  </Modal>
                </Portal>
                {/* End date picker with modal Ends */}

                <View>
                  <DropDown
                    label={t("Project Status")}
                    mode={"flat"}
                    visible={showProjectStatusDropdown}
                    showDropDown={() => setShowProjectStatusDropdown(true)}
                    onDismiss={() => setShowProjectStatusDropdown(false)}
                    value={values.status}
                    setValue={(data) => {
                      setFieldValue("status", data);
                    }}
                    inputProps={{
                      activeUnderlineColor: "#53555e",
                      backgroundColor: "#2a2c38",
                      underlineColor: "#53555e",
                      borderColor: "#53555e",
                      style: { backgroundColor: "#2a2c38" },
                      underlineStyle: {
                        borderWidth: 2,
                        borderColor: "#53555e",
                      },
                      right: (
                        <TextInput.Icon
                          icon={
                            showProjectStatusDropdown ? "menu-up" : "menu-down"
                          }
                          iconColor="white"
                        />
                      ),
                    }}
                    list={PROJECT_STATUS}
                    theme={{
                      colors: {
                        onSurfaceVariant: "white",
                      },
                    }}
                  />
                </View>

                <View className="flex flex-row w-full">
                  <View className="flex flex-col w-1/2">
                    <TouchableOpacity
                      onPress={() => {
                        setFieldValue("payment_check", !values.payment_check);
                      }}
                      className="flex flex-row w-fit items-center justify-start"
                    >
                      <Checkbox.Android
                        status={values.payment_check ? "checked" : "unchecked"}
                        color="white"
                        underlayColor="white"
                      />
                      <Text className="text-white">{t("Payment Check")}</Text>
                    </TouchableOpacity>
                  </View>
                  <View className="flex flex-col w-1/2">
                    <TouchableOpacity
                      onPress={() => {
                        setFieldValue("reporting_file", !values.reporting_file);
                      }}
                      className="flex flex-row w-fit items-center justify-start"
                    >
                      <Checkbox.Android
                        status={values.reporting_file ? "checked" : "unchecked"}
                        color="white"
                        underlayColor="white"
                      />
                      <Text className="text-white">{t("Reporting File")}</Text>
                    </TouchableOpacity>
                  </View>
                </View>

                <View className="flex flex-row w-full">
                  <View className="flex flex-col w-1/2">
                    <TouchableOpacity
                      onPress={() => {
                        setFieldValue("check_planning", !values.check_planning);
                      }}
                      className="flex flex-row w-fit items-center justify-start"
                    >
                      <Checkbox.Android
                        status={values.check_planning ? "checked" : "unchecked"}
                        color="white"
                        underlayColor="white"
                      />
                      <Text className="text-white">{t("Check Planning")}</Text>
                    </TouchableOpacity>
                  </View>
                  <View className="flex flex-col w-1/2">
                    <TouchableOpacity
                      onPress={() => {
                        setFieldValue("quality_plan", !values.quality_plan);
                      }}
                      className="flex flex-row w-fit items-center justify-start"
                    >
                      <Checkbox.Android
                        status={values.quality_plan ? "checked" : "unchecked"}
                        color="white"
                        underlayColor="white"
                      />
                      <Text className="text-white">{t("Quality Plan")}</Text>
                    </TouchableOpacity>
                  </View>
                </View>

                <View className="flex flex-row w-full">
                  <View className="flex flex-col w-1/2">
                    <TouchableOpacity
                      onPress={() => {
                        setFieldValue(
                          "expeses_to_be_reimbursed",
                          !values.expeses_to_be_reimbursed
                        );
                      }}
                      className="flex flex-row w-fit items-center justify-start"
                    >
                      <Checkbox.Android
                        status={
                          values.expeses_to_be_reimbursed
                            ? "checked"
                            : "unchecked"
                        }
                        color="white"
                        underlayColor="white"
                      />
                      <Text className="text-white">
                        {t("Expeses To Be Reimbursed")}
                      </Text>
                    </TouchableOpacity>
                  </View>

                  <View className="flex flex-col w-1/2">
                    <TouchableOpacity
                      onPress={() => {
                        setFieldValue(
                          "expeses_included",
                          !values.expeses_included
                        );
                      }}
                      className="flex flex-row w-fit items-center justify-start"
                    >
                      <Checkbox.Android
                        status={
                          values.expeses_included ? "checked" : "unchecked"
                        }
                        color="white"
                        underlayColor="white"
                      />
                      <Text className="text-white">
                        {t("Expenses Included")}
                      </Text>
                    </TouchableOpacity>
                  </View>
                </View>

                <View className="py-2 w-full items-center justify-center">
                  <Button
                    mode="contained"
                    onPress={handleSubmit as (e: unknown) => void}
                    className="p-2"
                    buttonColor="#ffdf6b"
                  >
                    {loading || editProjectLoading ? (
                      <ActivityIndicator animating={true} color={"red"} />
                    ) : (
                      <Text className="text-primary font-bold">
                        Save Project
                      </Text>
                    )}
                  </Button>
                </View>
              </View>
            )}
          </Formik>
        </View>
      </View>
    </ScrollView>
  );
}

export default CreateNewProject;
