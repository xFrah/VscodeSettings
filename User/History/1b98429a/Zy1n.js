import { createStore } from 'vuex';

export default createStore({
    state: {
        websocketData: null,
    },
    mutations: {
        setWebsocketData(state, data) {
            state.websocketData = data;
            console.log(data);
        },
    },
    actions: {
        initializeWebsocket({ commit }) {
            console.log('initializeWebsocket');
            const ws = new WebSocket('ws://localhost:3001/ws');
            console.log(ws);
            ws.onopen = () => {
                console.log('Connected to the WebSocket server');
            };
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            ws.onmessage = (event) => {
                commit('setWebsocketData', JSON.parse(event.data));
            };

            // Aggiungi gestione degli errori e dell'evento onclose come necessario
        },
    },
});
