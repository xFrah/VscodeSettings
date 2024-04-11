cd "C:\Program Files\VideoLAN\VLC"


rem VLc session 2 with gigaport2 audio, port 8320, and russound2$

start cmd.exe /k "VLC --http-port 8090 --audio-track=1 --no-video --directx-audio-device={58527C7E-93EB-459D-BE75-D8D08D3190D3} "C:\Users\DadoR\Videos\The.Italian.Job.2003.BDRip.720p.x264.AC3.ITA-ENG.mkv" 

rem VLC session 1 with gigaport1 audio, port 8310, and russound1$

start cmd.exe /k "VLC --http-port 8091 --audio-track=0  --fullscreen --directx-audio-device={E0B0541A-E424-4F0F-A01B-210009BC4862} "C:\Users\DadoR\Videos\The.Italian.Job.2003.BDRip.720p.x264.AC3.ITA-ENG.mkv" 



