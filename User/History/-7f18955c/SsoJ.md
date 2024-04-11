0. Installa AnyDesk 
   1. Apri Qterminal ed esegui il comando "sudo su"
   2. Apri questo link su firefox ed esegui i comandi per installare anydesk: http://deb.anydesk.com/howto.html
   3. Apri AnyDesk dal menu ed invia a Fra l'id della macchina
   4. Apri le impostazioni in alto a destra
   5. Apri la voce Sicurezza (a sinistra)
   6. Sblocca impostazioni di sicurezza (in alto)
   7. Abilita Accesso non vigilato, con password tvm2023! e profilo "Accesso completo"
   8. Mandami un messaggio e dimmi di provare a connettermi.
1. Installa la repo
   1. Copia TotemPython sul desktop
   2. Prendi il terminal id dal POS (GREVE)
   3. Modifica config.json in TotemPython con la giusta configurazione (cambia topic e terminal_id)
   4. Installa paho-mqtt con: "pip install paho-mqtt"
   5. Prova a runnare il main con: "python3 main.py"
2. Configura l'autostart
   1. Copia starter.sh sul desktop
   2. Dagli i permessi di esecuzione con: sudo chmod +x starter.sh (devi essere sul desktop col cmd, "cd Desktop")
   3. sudo nano /etc/xdg/autostart/tvm.desktop
   4. Copia il contenuto di tvm.desktop (nostro) nel terminale
   5. CTRL + x per chiudere, poi y per salvare le modifiche

Se il POS non funziona va riavviato tutto, stacca e riattacca la corrente generale.
Il problema è che una volta che stabilisce una connessione e viene interrotta, non può avviarne un'altra.