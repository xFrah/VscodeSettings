<template>
  <div id="app">
    <MapComponent />
    <ToolbarComponent />
  </div>
</template>

<script>
import { defineComponent } from 'vue';
import MapComponent from './components/MapComponent.vue';
import ToolbarComponent from './components/ToolbarComponent.vue';

export default defineComponent({
  name: 'App',
  components: {
    MapComponent,
    ToolbarComponent
  },
  mounted() {
    // this.$store.dispatch('initializeWebsocket');
  }
});
</script>

<style>
/* Aggiungi qui gli stili globali */
</style>
