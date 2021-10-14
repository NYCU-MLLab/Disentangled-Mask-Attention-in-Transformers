# Disentangled-Transformer-Demo

1. Install
```
> cd web
> npm install
```

2. Run web interface
```
> cd web
> npm start
```

> Port can be changed in `web/package.json`.
> IP of web server can be changed in `web/src/home/configs.js`

> WebAPI for machine translation follows following format (GET & POST methods):  
> http://<server_ip>/inference/mt/<task_name>/<src_sentence>  
> Supported <task_name>: DeEn, EnDe, ZhEn

> WebAPI for speech recognition follows following format (POST method only):  
> http://<server_ip>/inference/asr/

3. Run web server
```
> cd inference server
> python3 server.py
```

> Path of checkpoint for machine translation (Fairseq) and speech recognition (SpeechBrain) can be modified in `inference_server/configs.json`