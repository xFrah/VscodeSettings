import React, { useState } from "react";
import { Chart } from "react-google-charts";
import { View } from "react-native";
import { Button } from "react-native-paper";
import { APP_CURRENCY } from "../constants/App.constants";

function AnalysisCharts({ ProjectData }) {
  const data = [
    ["Cost", "Amount"],
    // [
    //   "Expences",
    //   ProjectData?.getCostMarginBudgetDashboardDetails?.totalCostOfExpences ||
    //     0,
    // ],
    [
      "Worker Cost",
      ProjectData?.getCostMarginBudgetDashboardDetails?.totalCostOfWorker || 0,
    ],
    [
      "Receipt's",
      ProjectData?.getCostMarginBudgetDashboardDetails?.totalValueOfReceipt ||
        0,
    ],
  ];

  const data2 = [
    ["Total Budget", "Fee", "Fixed Cost", "Cost"],
    [
      `${ProjectData?.getCostMarginBudgetDashboardDetails?.name}`,
      ProjectData?.getCostMarginBudgetDashboardDetails?.fee || 0,
      ProjectData?.getCostMarginBudgetDashboardDetails?.margin || 0,
      ProjectData?.getCostMarginBudgetDashboardDetails?.cost || 0,
    ],
  ];

  const [activeTab, setActiveTab] = useState(1);

  const handleTabClick = (tabNumber) => {
    setActiveTab(tabNumber);
  };

  return (
    <>
      <View className="flex flex-col w-full my-4 border border-white">
        <View className="flex flex-row w-full">
          <Button
            mode={activeTab === 1 ? "contained" : "outlined"}
            buttonColor={activeTab === 1 ? "#3b8af7" : null}
            textColor="white"
            style={{ borderRadius: 0 }}
            className="w-2/12 border border-white"
            onPress={() => handleTabClick(1)}
          >
            Overview costi
          </Button>
        </View>
        <View className="flex flex-row w-full">
          <View className="flex flex-row w-2/3">
          <Chart
              chartType="BarChart"
              width="100%"
              height="400px"
              data={data2}
              options={{
                title: `Total Budget of Project ${ProjectData?.getCostMarginBudgetDashboardDetails?.budget} ${APP_CURRENCY}\nMargin ${fixedCost} ${APP_CURRENCY}\nOver Budget ${overBudgetCost} ${APP_CURRENCY}`,
                chartArea: { width: "70%" },
                isStacked: true,
                backgroundColor: "#2A2C38",
                titleTextStyle: { color: "white", fontSize: 15 },
                legend: { textStyle: { color: "white" }, position: "bottom" },
                hAxis: { textStyle: { color: "white" } },
                colors: colors,
                vAxis: { textStyle: { color: "white" } },
              }}
            />
          </View>
          {ProjectData?.getCostMarginBudgetDashboardDetails?.cost ? (
            <View className="flex flex-col w-1/3">
              <Chart
                chartType="PieChart"
                data={data}
                options={{
                  title: "Distribution of Cost",
                  titleTextStyle: { color: "white", fontSize: 15 },
                  colors: ['#00aac8', "#ffa256","#69c474"],
                  backgroundColor: "#2A2C38",
                  legend: {
                    textStyle: { color: "white" },
                    position: "bottom",
                  },
                  pieSliceText: 'value-and-percentage',
                }}
                width={"100%"}
                height={"100%"}
              />
            </View>
          ) : null}
        </View>
      </View>
    </>
  );
}

export default AnalysisCharts;
