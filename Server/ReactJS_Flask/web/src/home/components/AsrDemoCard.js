import React, { useState } from 'react'
import {Button, Typography, Card, Divider} from '@material-ui/core'
import BaseCard from './BaseCard';
import DemoSteps from './DemoSteps';
import {configs} from '../configs';
import AttentionTable from './AttentionTable';

function ASRUploadButton(props) {
    return (
        <div>
            <input type="file"
                id="asr_upload_input"
                onChange={(e) => props.handleASRFileUpload(e.target.files)}
                style={{display: "none"}}/>   
            <Typography>{props.audioFilename}</Typography>        
            <Button variant="contained" color="#757575" style={{marginRight: 15}}
                    onClick={() => document.getElementById("asr_upload_input").click()}>
                Upload
            </Button>
                {
                    props.audio.length > 0 ? 
                    <Button variant="contained"  color="#757575" style={{marginRight: 15}}
                            onClick={() => new Audio(props.audio).play()}>
                        Play
                    </Button> : null
                }
        </div>
    );
}

function InputContent(audio, audioFilename, handleASRFileUpload) {
    return(
        <ASRUploadButton
            audio={audio}
            audioFilename={audioFilename}
            handleASRFileUpload={handleASRFileUpload}
        />
    );
}

function ResultContent(
    inferenceFinish, results, reInferenceModel, head, layer, setHead, setLayer
) {
    return (
        <div>
            <Card style={{marginBottom: 30}} elevation={0}>
                <Typography>Recognition result</Typography>
                <Divider/>
                <Typography style={{margin: 20}}>{results["result"]}</Typography>
            </Card>
            <AttentionTable
                inferenceFinish={inferenceFinish}
                inferenceResult={results}
                reInferenceModel={reInferenceModel}
                head={head}
                layer={layer}
                setHead={setHead}
                setLayer={setLayer}
            />
        </div>
    );
}

function AsrDemoCard() {
    const [audio, setAudio] = useState("");
    const [audioFilename, setAudioFilename] = useState("");
    const [layer, setLayer] = useState(6);
    const [head, setHead] = useState(4);

    async function handleASRFileUpload(files) {
        const fileReader = new FileReader();
        fileReader.readAsDataURL(files[0]);
        fileReader.onload = (e) => setAudio(e.target.result);
        setAudioFilename(files[0].name);
    }

    return (
        <BaseCard title="Demo">
            <DemoSteps
                steps={['Upload speech file', 'Inference model', 'View results']}
                data={() => ({"audio": audio, "layer": layer-1, "head": head-1})}
                serverIp={() => configs["ASR_inference_server"]+"inference/asr/"}                    
                inputContent={() => InputContent(
                    audio, audioFilename, handleASRFileUpload
                )}
                resultContent={(inferenceFinish, results, reInferenceModel) => ResultContent(
                    inferenceFinish, results, reInferenceModel, head, layer, setHead, setLayer
                )}
            />
        </BaseCard>
    );
}

export default AsrDemoCard;