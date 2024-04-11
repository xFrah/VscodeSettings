<template>
  <div id="map-container">
    <div id="legend" class="legend">
      <div id="color-gradient"></div>
      <div class="legend-scale">
        <span id="min-value">Min Value</span>
        <span id="max-value">Max Value</span>
      </div>
    </div>
  </div>
  <div id="tooltip" class="ol-tooltip"></div>
</template>

<script>
import Map from 'ol/Map';
import { mapState } from 'vuex';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import XYZ from 'ol/source/XYZ';
import { fromLonLat } from 'ol/proj';
import { boundingExtent } from 'ol/extent';
import Feature from 'ol/Feature';
import Point from 'ol/geom/Point';
import { Circle as CircleStyle, Fill, Style } from 'ol/style';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import Overlay from 'ol/Overlay';
import Icon from 'ol/style/Icon';
import LineString from 'ol/geom/LineString';
import Polygon from 'ol/geom/Polygon';
import Stroke from 'ol/style/Stroke';
import IDW from 'ol-ext/source/IDW';
import Crop from 'ol-ext/filter/Crop';
import Image from 'ol/layer/Image';
import { defaults as defaultInteractions } from 'ol/interaction';

export default {
  data() {
    return {
      table: this.generateRandomIntegers(100),
      limitedWaypointFeatures: [],
      droneFeature: null,
      waypoints: null,
      waypointLayer: null,
      waypointFeatures: null,
      pathLayer: null,
      //first layer heatmap
      humidityIdwLayer: null,
      humidityImageLayer: null,
      //second layer heatmap
      tempIdwLayer: null,
      tempImageLayer: null,
      layerIndex: 0,
      idwImageLayers: [],
    };
  },

  name: 'RotatedGoogleMapsLayer',
  async mounted() {
    const response = await fetch('dati.json');
    if (!response.ok) {
      throw new Error('Network response was not ok ' + response.statusText);
    }
    const json = await response.json();
    const coords = this.getCoordinates(json);
    const center = this.getCenter(json);

    const customInteractions = defaultInteractions({ doubleClickZoom: false });
    this.updateLegend('none');

    const map = new Map({
      target: 'map-container',
      controls: [],
      interactions: customInteractions, // Use custom interactions here
      layers: [
        new TileLayer({
          source: new XYZ({
            url: 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attributions: [
              '&copy; <a href="https://maps.google.com">Google Maps</a>'
            ],
          }),
        })
      ],
      view: new View({
        zoom: 18,
        rotation: this.calculateRotationAngle(coords),
        center: center, // Coordinate dello Stadio Olimpico di Roma
      })
    });
    map.on('dblclick', this.handleMapDoubleClick);
    // Calcolo dell'estensione che include tutti i marker
    const extent = boundingExtent(coords);
    map.getView().fit(extent, { padding: [50, 50, 50, 50] });

    // Creazione dei waypoint
    const waypoints = [

      { id: 1, coor: [12.454682, 41.934221], color: 'grey', val: 20 },
      { id: 2, coor: [12.454690, 41.933664], color: 'grey', val: 40 },
      { id: 3, coor: [12.455217, 41.933617], color: 'grey', val: 90 }
      // Altri waypoints
    ];

    this.waypointFeatures = waypoints.map(wp => {
      // generate random value from 0 to 100
      wp.val = Math.random() * 100;
      const feature = new Feature({
        geometry: new Point(fromLonLat(wp.coor)),
        id: wp.id,
        val: wp.val
      });

      feature.setStyle(new Style({
        image: new CircleStyle({
          radius: 10,
          fill: new Fill({ color: wp.color }),
          stroke: new Stroke({
            color: 'white',
            width: 1.5
          })
        })
      }));

      return feature;
    });

    this.waypointLayer = new VectorLayer({
      source: new VectorSource({
        features: this.waypointFeatures
      })
    });

    // Creazione del tooltip
    const tooltipElement = document.getElementById('tooltip');
    const tooltipOverlay = new Overlay({
      element: tooltipElement,
      offset: [10, 0],
      positioning: 'bottom-left'
    });

    // Mostra il tooltip al passaggio del mouse
    map.on('pointermove', (event) => {
      if (map.hasFeatureAtPixel(event.pixel)) {
        const features = map.getFeaturesAtPixel(event.pixel);
        const isDroneFeature = features.some(feature => feature === this.droneFeature);
        const isPathFeature = features.some(feature => feature === pathFeature);

        if (!isDroneFeature && !isPathFeature) {
          // Mostra il tooltip solo se la feature non è il drone e il path
          const feature = features[0];
          tooltipOverlay.setPosition(event.coordinate);
          tooltipElement.innerHTML = `Waypoint ID: ${feature.get('id')}<br>Umidità: ${feature.get('umidità')}`;
          tooltipElement.style.display = '';
        } else {
          // Nascondi il tooltip se la feature è il drone
          tooltipElement.style.display = 'none';
        }
      } else {
        tooltipElement.style.display = 'none';
      }
    });

    // Creazione del drone immagine
    this.droneFeature = new Feature({
      geometry: new Point(fromLonLat([12.454764, 41.933961]))
    });

    this.droneFeature.setStyle(new Style({
      image: new Icon({
        src: require('../assets/drone.png'),
        scale: 0.5 // Scala l'immagine se necessario
      })
    }));

    const droneLayer = new VectorLayer({
      source: new VectorSource({
        features: [this.droneFeature]
      })
    });

    // Creazione del percorso
    const shortestPath = this.calculateShortestPath(waypoints);
    const pathGeometry = new LineString(shortestPath.map(p => fromLonLat(p.coor)));
    const pathFeature = new Feature(pathGeometry);

    this.pathLayer = new VectorLayer({
      source: new VectorSource({
        features: [pathFeature]
      }),
      style: new Style({
        stroke: new Stroke({
          color: '#0583d2',
          width: 2
        })
      })
    });

    this.humidityIdwLayer = new IDW({
      useWorker: true,
      getColor: function (value) {
        // Define start (light blue) and end (orange) colors in RGB
        const endColor = { r: 173, g: 216, b: 230 }; // Light blue
        const startColor = { r: 255, g: 165, b: 0 }; // Orange

        // Normalize the value to be between 0 and 1
        // Assuming 'value' ranges from 0 to 100 (adjust according to your data range)
        const normalizedValue = Math.min(Math.max(value / 100, 0), 1);

        // Interpolate between the start and end colors
        const r = Math.round(startColor.r + (endColor.r - startColor.r) * normalizedValue);
        const g = Math.round(startColor.g + (endColor.g - startColor.g) * normalizedValue);
        const b = Math.round(startColor.b + (endColor.b - startColor.b) * normalizedValue);

        // Return the color as an RGBA array
        return [r, g, b, 255]; // 255 is the alpha value (fully opaque)
      },
      /**/
      // scale: 8,
      // Source that contains the data
      source: new VectorSource(),
      // Use val as weight property
      weight: 'val'

    });
    this.humidityIdwLayer.getSource().addFeatures(this.waypointFeatures);

    const humidityImageLayer = new Image({
      title: 'IDW',
      source: this.humidityIdwLayer,
      opacity: .5,
      visible: false // Initially hidden
    });

    // Add the Image layer to the array
    this.idwImageLayers.push(humidityImageLayer);

    const tempIdwLayer = new IDW({
      useWorker: true,
      getColor: function (value) {
        // Define the colors for the thermal gradient
        const colors = [
          { r: 0, g: 0, b: 255 },   // Blue
          { r: 0, g: 255, b: 0 },   // Green
          { r: 255, g: 255, b: 0 }, // Yellow
          { r: 255, g: 0, b: 0 },   // Red
          { r: 255, g: 255, b: 255 } // White
        ];

        // Normalize the value to be between 0 and 1
        // Assuming 'value' ranges from minTemp to maxTemp
        const minTemp = 0; // Adjust as needed
        const maxTemp = 100; // Adjust as needed
        const normalizedValue = (value - minTemp) / (maxTemp - minTemp);

        // Determine which two colors to interpolate between
        const index = Math.floor(normalizedValue * (colors.length - 1));
        const color1 = colors[index];
        const color2 = colors[Math.min(index + 1, colors.length - 1)];

        // Calculate interpolation factor between the two colors
        const factor = (normalizedValue * (colors.length - 1)) - index;

        // Interpolate between the two colors
        const r = Math.round(color1.r + (color2.r - color1.r) * factor);
        const g = Math.round(color1.g + (color2.g - color1.g) * factor);
        const b = Math.round(color1.b + (color2.b - color1.b) * factor);

        // Return the color as an RGBA array
        return [r, g, b, 255]; // 255 is the alpha value (fully opaque)
      },

      /**/
      // scale: 8,
      // Source that contains the data
      source: new VectorSource(),
      // Use val as weight property
      weight: 'val'

    });
    tempIdwLayer.getSource().addFeatures(this.waypointFeatures);

    const TempImageLayer = new Image({
      title: 'IDW',
      source: tempIdwLayer,
      opacity: .5,
      visible: false // Initially hidden
    });

    // Add the Image layer to the array
    this.idwImageLayers.push(TempImageLayer);

    const polygonFeature = new Feature(new Polygon([coords]));

    // Create and apply the crop filter to the IDW layer
    const cropFilter = new Crop({
      feature: polygonFeature,
      inner: false // Change to true to crop the inner part of the polygon
    });
    // for each layer in the array, add the filter
    this.idwImageLayers.forEach(idwImageLayer => idwImageLayer.addFilter(cropFilter));
    //idwImageLayer.addFilter(cropFilter);

    map.addLayer(humidityImageLayer);
    map.addLayer(TempImageLayer);
    map.addLayer(this.pathLayer);
    map.addLayer(this.waypointLayer);
    map.addLayer(droneLayer);
    map.addOverlay(tooltipOverlay);
  },
  computed: {
    ...mapState(['websocketData']),
  },
  watch: {
    websocketData(newData) {
      if (newData) {
        if (!this.droneFeature || !this.waypointFeatures) {
          return;
        }
        // Assumi che newData abbia le proprietà 'lat' e 'lon' per le coordinate
        const newCoordinates = fromLonLat([newData.lon, newData.lat]);
        this.droneFeature.getGeometry().setCoordinates(newCoordinates);
        // newdata ha field waypoints, forse
        // check that waypoints exists in newData
        console.log(1);
        if (newData.markers) {
          const waypoints = newData.markers;  // waypoints is an array of tuples lat lon
          waypoints.forEach(wp => {
            wp.geometry = new Point(fromLonLat(wp.lon, wp.lat));
          });
          console.log(2);
          this.waypoints = waypoints;
          var i = 0;
          const waypointFeatures = waypoints.map(wp => {
            const feature = new Feature({
              geometry: wp.geometry,
              id: wp.id,
            });
            console.log(3);
            // if i < newData.current_waypoint, set the color to green
            const color = i < newData.current_waypoint ? 'green' : 'grey';

            feature.setStyle(new Style({
              image: new CircleStyle({
                radius: 7,
                fill: new Fill({ color: color })
              })
            }));
            i++;
            return feature;
          });
          this.waypointLayer.getSource().clear();
          this.waypointLayer.getSource().addFeatures(waypointFeatures);

          this.pathLayer.getSource().clear();
          const pathGeometry = new LineString(waypoints.map(wp => fromLonLat([wp.lon, wp.lat])));
          const pathFeature = new Feature(pathGeometry);
          this.pathLayer.getSource().addFeature(pathFeature);

          // Update the IDW layer with the new data
          this.humidityIdwLayer.getSource().clear();
          // get version of waypoints until current_waypoint
          const currentWaypoint = newData.current_waypoint;
          // check if waypointfeature length has changed
          const waypointsUntilCurrent = waypoints.slice(0, currentWaypoint + 1);
          this.humidityIdwLayer.getSource().addFeatures(waypointsUntilCurrent.map(wp => {
            const feature = new Feature({
              geometry: wp.geometry,
              val: this.table[wp.id],
            });
            return feature;
          }));
        }
      }
    },
  },

  methods: {
    handleMapDoubleClick(event) {
      // Toggle the visibility of the heatmap layer
      console.log(event);
      // Hide all IDW layers first
      this.idwImageLayers.forEach(layer => layer.setVisible(false));

      // Increment layerIndex and wrap around the length of idwImageLayers
      this.layerIndex = (this.layerIndex + 1) % (this.idwImageLayers.length + 1);
      console.log(this.layerIndex);
      // Show the layer corresponding to the current index, if layerIndex is not 0
      if (this.layerIndex !== 0) {
        this.idwImageLayers[this.layerIndex - 1].setVisible(true);
        this.waypointLayer.setVisible(false);
        this.pathLayer.setVisible(false);
      } else {
        this.waypointLayer.setVisible(true);
        this.pathLayer.setVisible(true);
      }
      // ... any additional logic for double-click ...
      const legend = document.getElementById('legend');
      // Check which layer is currently visible and update the legend accordingly
      if (this.layerIndex === 1) { // Assuming 1 is the index for humidity layer
        this.updateLegend('humidity');
        legend.style.display = 'block';
      } else if (this.layerIndex === 2) { // Assuming 2 is the index for temperature layer
        this.updateLegend('temperature');
        legend.style.display = 'block';
      } else if (this.layerIndex === 0) {
        // Hide or reset the legend when no layer is visible
        legend.style.display = 'none';
      }
    },
    generateRandomIntegers(length) {
      const randomIntegers = [];
      for (let i = 0; i < length; i++) {
        // Generate a random integer between 0 and 100
        const randomInt = Math.floor(Math.random() * 101);
        randomIntegers.push(randomInt);
      }
      return randomIntegers;
    },
    updateLegend(layerType) {
      const minLabel = document.getElementById('min-value');
      const maxLabel = document.getElementById('max-value');
      const gradient = document.getElementById('color-gradient');

      if (layerType === 'humidity') {
        minLabel.innerText = 'Low Humidity';
        minLabel.style.fontSize = '22px';
        maxLabel.innerText = 'High Humidity';
        maxLabel.style.fontSize = '22px';
        gradient.style.background = 'linear-gradient(to right, orange, lightblue)';
        gradient.style.width = '20vw';
      } else if (layerType === 'temperature') {
        minLabel.innerText = 'Cool';
        minLabel.style.fontSize = '22px';
        maxLabel.innerText = 'Hot';
        maxLabel.style.fontSize = '22px';
        gradient.style.background = 'linear-gradient(to right, blue, green, yellow, red, white)';
        gradient.style.width = '20vw';
      }
    },
    hue2rgb(h) {
      h = (h + 6) % 6;
      if (h < 1) return Math.round(h * 255);
      if (h < 3) return 255;
      if (h < 4) return Math.round((4 - h) * 255);
      return 0;
    },
    calculateRotationAngle(coords) {
      // Ensure the coordinates are in the correct format
      if (coords.length < 2 || !Array.isArray(coords[0]) || !Array.isArray(coords[1])) {
        throw new Error('Invalid coordinates');
      }

      // Get the first two points
      const point1 = coords[0];
      const point2 = coords[1];

      // Calculate the difference in coordinates
      const dx = point2[0] - point1[0];
      const dy = point2[1] - point1[1];

      // Calculate the angle in radians
      const angle = Math.atan2(dy, dx);

      return angle;
    },
    getCoordinates(jsonData) {
      let data;
      if (typeof jsonData === 'string') {
        data = JSON.parse(jsonData);
      } else {
        data = jsonData;
      }

      // Check if the "coordinates" key exists
      if (!data.coordinates || !Array.isArray(data.coordinates)) {
        throw new Error('Invalid or missing coordinates in JSON data');
      }

      // Convert each pair of coordinates using fromLonLat
      const coords = data.coordinates.map(coordPair => {
        if (coordPair.length === 2) {
          return fromLonLat([coordPair[0], coordPair[1]]); // Switch lat and lon if necessary
        } else {
          throw new Error('Invalid coordinate pair');
        }
      });
      return coords;
    },
    getCenter(jsonData) {
      let data;
      if (typeof jsonData === 'string') {
        data = JSON.parse(jsonData);
      } else {
        data = jsonData;
      }

      // Check if the "center" key exists
      if (!data.center || !Array.isArray(data.center)) {
        throw new Error('Invalid or missing center in JSON data');
      }

      // Convert the center coordinates using fromLonLat
      const center = fromLonLat([data.center[0], data.center[1]]); // Switch lat and lon if necessary
      return center;
    },
    haversineDistance(coord1, coord2) {
      const toRad = x => (x * Math.PI) / 180;
      const R = 6371; // Raggio della Terra in chilometri
      const dLat = toRad(coord2[0] - coord1[0]);
      const dLon = toRad(coord2[1] - coord1[1]);
      const lat1 = toRad(coord1[0]);
      const lat2 = toRad(coord2[0]);

      const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.sin(dLon / 2) * Math.sin(dLon / 2) * Math.cos(lat1) * Math.cos(lat2);
      const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
      return R * c; // Distanza in chilometri
    },
    calculateShortestPath(waypoints) {
      let minDistance = Number.MAX_VALUE;
      let shortestPath = [];

      // Genera tutte le permutazioni dei waypoints
      const permute = (arr, m = []) => {
        if (arr.length === 0) {
          // Calcola la distanza totale per questa permutazione
          let distance = this.calculatePathDistance(m);
          if (distance < minDistance) {
            minDistance = distance;
            shortestPath = m.slice();
          }
        } else {
          arr.forEach((e, i) => {
            let curr = arr.slice();
            let next = curr.splice(i, 1);
            permute(curr.slice(), m.concat(next));
          });
        }
      };

      permute(waypoints);

      return shortestPath;
    },
    calculatePathDistance(path) {
      let distance = 0;
      for (let i = 0; i < path.length - 1; i++) {
        distance += this.haversineDistance(path[i].coor, path[i + 1].coor);
      }
      return distance;
    },

    // updateWaypointColor(waypointId) {
    //   const feature = this.waypointFeatures.find(f => f.get('id') === waypointId);
    //   if (feature) {
    //     feature.getStyle().getImage().getFill().setColor('green');
    //   }
    // }
  }
};
</script>

<style>
#map-container {
  position: relative;
  width: 100%;
  height: 96vh;
}

.ol-tooltip {
  position: absolute;
  background-color: white;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
  padding: 4px 8px;
  border-radius: 6px;
  border: 1px solid #cccccc;
  pointer-events: none;
  opacity: 0.7;
  transition: opacity 0.3s;
}

.legend {
  display: none;
  position: absolute;
  top: 10px;
  /* Adjust as needed */
  left: 10px;
  /* Adjust as needed */
  background: white;
  padding: 10px;
  border-radius: 4px;
  z-index: 10;
  /* Ensure it's above the map */
}

#color-gradient {
  width: 100%;
  height: 20px;
  border: 1px solid #000;
  background: linear-gradient(to right, #00BFFF, orange);
  /* Replace with your gradient */
}

.legend-scale {
  display: flex;
  justify-content: space-between;
  font-size: 18px;
  margin-top: 5px;
}

#min-value {
  margin-right: 10px;
}

#max-value {
  margin-left: 10px;
}
</style>
