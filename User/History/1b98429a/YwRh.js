import { createStore } from 'vuex';

export default createStore({
    state: {
        websocketData: null,
    },
    mutations: {
        setWebsocketData(state, data) {
            state.websocketData = data;
        },
    },
    actions: {
        initializeWebsocket({ commit }) {
            const ws = new WebSocket('ws://YOUR_SERVER_ADDRESS');

            ws.onmessage = (event) => {
                commit('setWebsocketData', JSON.parse(event.data));
            };

            // Aggiungi gestione degli errori e dell'evento onclose come necessario
        },
    },
});
