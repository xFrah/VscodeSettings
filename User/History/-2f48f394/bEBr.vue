<template>
  <div class="toolbar">
    <h3 class="title">Live Data</h3>
    <div class="altitudeChart">
      <p class="text">Altitude</p>
      <apexchart ref="altitudeChart" type="line" :options="altitudeChartOptions" :series="altitudeSeries"></apexchart>
    </div>
    <div class="resultChart">
      <p class="text-result">Results</p>
      <!--<apexchart ref="conductivityChart" type="line" :options="conductivityChartOptions" :series="conductivitySeries">-->
      <apexchart ref="resultChart" type="radar" :options="resultChartOptions" :series="resultSeries"></apexchart>
    </div>
    <div class="battery-status">
      <p class="text">Battery</p>
      <p class="text-drone-status">{{ batteryLevel }}%</p>
      <!-- Add dynamic battery icon here -->
    </div>
    <div class="drone-status">
      <p class="text">Status:</p>
      <p class="text-drone-status">{{ droneStatus }}</p>
    </div>
  </div>
</template>

<script>
import VueApexCharts from 'vue3-apexcharts'
import { mapState } from 'vuex';

export default {
  name: 'ToolbarComponent',
  components: {
    apexchart: VueApexCharts,
  },
  data() {
    return {
      batteryLevel: 100, // Example battery level
      droneStatus: 'Idle', // Example drone status
      altitudeChartOptions: {
        chart: {
          id: 'altitudeChart',
          type: 'line',
          sparkline: {
            enabled: false
          },
          toolbar: {
            autoSelected: 'pan',
            show: false
          },
          height: 350,
          zoom: {
            enabled: false
          },
        },
        xaxis: {
          show: false,
          labels: {
            show: false // Add this line to hide the x-axis labels/numbers
          }
        },
        yaxis: {
          show: true,
          min: 0,
          max: 20,
        },
        stroke: {
          curve: 'smooth',
          colors: ['#0583d2'],
        },
        markers: {
          size: 0
        },
      },
      altitudeSeries: [{
        name: 'Altitude',
        data: [] // Your altitude data here
      }],
      conductivityChartOptions: {
        chart: {
          id: 'conductivityChart',
          type: 'line',
          sparkline: {
            enabled: true
          },
          toolbar: {
            autoSelected: 'pan',
            show: false
          },
          height: 350,
          zoom: {
            enabled: false
          },
        },
        xaxis: {
          show: false,
          labels: {
            show: true // Add this line to hide the x-axis labels/numbers
          }
        },
        yaxis: {
          show: true,
          min: 0,
          max: 150,
        },
        stroke: {
          curve: 'smooth',
          colors: ['#0583d2'],
        },
        dataLabels: {
          enabled: false
        },
        markers: {
          size: 0
        },
      },
      conductivitySeries: [{
        name: 'conductivity',
        data: []
      }],
      //implement a radar chart with multiple series for the results
      resultChartOptions: {
        chart: {
          id: 'resultChart',
          type: 'radar',
          dropShadow: {
            enabled: true,
            blur: 1,
            left: 1,
            top: 1
          },
          sparkline: {
            enabled: false
          },
          toolbar: {
            autoSelected: 'pan',
            show: false
          },
          height: 100,
          zoom: {
            enabled: false
          },
        },
        xaxis: {
          show: true,
          categories: ['Humidity', 'Conductivity', 'Temperature', 'pH', 'Oxygen'],
          labels: {
            show: true,
            style: {
              colors: ["black"],
              fontSize: "12px",
              fontFamily: 'Arial'
            }
          },
        },
        yaxis: {
          show: false,
        },
        stroke: {
          show: true,
          colors: ['#0583d2'],
        },
        dataLabels: {
          enabled: false
        },
        markers: {
          size: 0
        },
      },
      resultSeries: [{
        name: 'optimisticResult',
        data: [40, 20, 39, 48, 33]
      },
      {
        name: 'result',
        data: []
      }],
    };
  },
  // write call to updateConductivityChartData() in mounted() hook
  mounted() {
    this.conductivityChart = this.$refs.conductivityChart;
    this.updateConductivityChartData();
    console.log('this.conductivitySeries: ', this.conductivitySeries);
  },
  computed: {
    ...mapState(['websocketData']),
  },
  watch: {
    websocketData(newData) {
      if (newData) {
        // Update Altitude
        if (newData.alt !== undefined) {
          if (this.altitudeSeries[0].data.length > 20) {
            this.altitudeSeries[0].data.shift();
          }
          this.altitudeSeries[0].data.push(parseFloat(newData.alt.toFixed(2)));
          this.altitudeSeries = [...this.altitudeSeries];
        }
        const resHumidity = [];
        const resTemp = [];

        newData.markers.forEach((wp) => {
          if (wp.id < newData.current_waypoint) {
            if (wp.humidity !== undefined) {
              resHumidity.push(wp.humidity)
            }
            if (wp.temp !== undefined) {
              resTemp.push(wp.temp)
            }
          }

          const avgHumidity = resHumidity.reduce((a, b) => a + b, 0) / resHumidity.length;
          this.resultSeries[1].data[0].push(parseInt(avgHumidity));

          const avgTemp = resTemp.reduce((a, b) => a + b, 0) / resTemp.length;
          this.resultSeries[1].data[1].push(parseInt(avgTemp));

        })
        // Update Drone Status
        if (newData.status !== undefined) {
          this.droneStatus = newData.status;
        }
        //Update Conductivity chart
        if (newData.alt !== undefined && newData.status == 'MEASURING') {
          this.updateConductivityChartData();
        }
        // Update Conductivity (if you have a separate chart for this)
        // if (newData.conductivity !== undefined) {
        //   // Assuming you have a similar setup for conductivitySeries
        //   this.conductivitySeries[0].data.push(newData.conductivity);
        //   this.conductivitySeries = [...this.conductivitySeries];
        // }
      }
    },
  },
  methods: {
    updateChartData(newData) {
      if (this.chart) {
        this.chart.updateSeries([{
          name: 'Altitude',
          data: newData
        }]);
      }
    },
    updateConductivityChartData() {
      const randomValue = Math.floor(Math.random() * 101); // Generate a random value

      // Add the new value to the series
      this.conductivitySeries[0].data.push(randomValue);

      // Optionally, limit the number of data points in the chart
      if (this.conductivitySeries[0].data.length > 20) {
        this.conductivitySeries[0].data.shift();
      }
      // Ensure reactivity
      this.conductivitySeries = [...this.conductivitySeries];
    }
  }
};
</script>

<style>
.toolbar {
  width: 325px;
  position: fixed;
  right: 0;
  top: 0;
  height: 85vh;
  background-color: #f4f4f4;
  padding: 20px;
  box-shadow: -2px 0 5px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 35px;
  margin-right: 40px;
  border-radius: 6px;
  opacity: 0.9;
}

.altitudeChart {
  width: 80%;
}

.resultChart {
  width: 100%;
  height: 100%;
  margin-bottom: 10px;
}

.battery-status,
.drone-status {
  margin-bottom: 10px;
  display: flex;
  justify-content: space-between;
}

.title {
  font-size: 25px;
  font-weight: bold;
  margin-bottom: 10px;
}

.text {
  font-size: 20px;
  font-weight: bold;
  margin-bottom: 10px;
}

.text-result {
  font-size: 20px;
  font-weight: bold;
  margin-bottom: 10px;
  margin-left: 35px;
}

.text-drone-status {
  font-size: 20px;
  font-weight: bold;
  margin-bottom: 10px;
  margin-left: 15px;
  color: #0583d2;
}
</style>
