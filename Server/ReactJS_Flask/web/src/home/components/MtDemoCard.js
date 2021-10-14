import React, { useState } from 'react'
import {InputLabel, Select, Typography, Card, TextField, Divider, FormControl, MenuItem} from '@material-ui/core'

import DemoSteps from './DemoSteps';
import {configs} from '../configs';
import AttentionTable from './AttentionTable';
import BaseCard from './BaseCard';

function InputContent(lang, supportLang, setSentence, setLang, reset) {
    return (
        <div style={{width: "100%", display: "table", textAlign: "center"}}>
            <div style={{display: "table-row"}}>
                {/* Language selector */}
                <div style={{display: "table-cell", width:"15%", paddingRight: "2.5%"}}>
                    <FormControl style={{width: "100%"}}>
                        <InputLabel>Language</InputLabel>
                        <Select value={lang} style={{textAlign: "center"}} onChange={(event) => {
                                setLang(event.target.value);
                                reset()
                            }}>
                            {supportLang.map(i => (
                                <MenuItem value={i}>{i}</MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </div>
                {/* Sentence input */}
                <div style={{display: "table-cell", width:"75%", textAlign: "left", paddingRight: "5%"}}>
                    <form noValidate autoComplete={false}>
                        <TextField
                            multiline
                            id="mtText"
                            label={"Sentence ("+lang.split(" ")[0]+")"}
                            onChange={(event) => (setSentence(event.target.value))}
                            style={{width: "100%"}}
                        />
                    </form>
                </div>
            </div>
        </div>
    );
};

function ResultContent(
    inferenceFinish, results, reInferenceModel, src_lang, tgt_lang, src_sentence, head, layer, setHead, setLayer
) {
    return (
        <div>
            <Card style={{marginBottom: 30}} elevation={0}>
                <Typography>Source sentence ({src_lang})</Typography>
                <Divider/>
                <Typography style={{margin: 20}}>{src_sentence}</Typography>
            </Card>
            <Card style={{marginBottom: 30}} elevation={0}>
                <Typography>Translated sentence ({tgt_lang})</Typography>
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

function MtDemoCard() {
    const [lang, setLang] = useState("German to English");
    const supportLang = ["German to English", "Chinese to English", "English to German"];
    const langToAPI = {
        "German to English": "DeEn",
        "English to German": "EnDe",
        "Chinese to English": "ZhEn"
    }
    const [sentence, setSentence] = useState("");
    const [layer, setLayer] = useState(6);
    const [head, setHead] = useState(4);

    return (        
        <BaseCard title="Demo">
            <DemoSteps
                steps={['Enter sentence', 'Inference model', 'View results']}
                data={() => ({"sentence": sentence, "layer": layer-1, "head": head-1})}
                serverIp={() => configs["MT_inference_server"]+"inference/mt/"+langToAPI[lang]+"/"}                    
                inputContent={(reset) => InputContent(lang, supportLang, setSentence, setLang, reset)}
                resultContent={(inferenceFinish, results, reInferenceModel) => ResultContent(
                    inferenceFinish, results, reInferenceModel, lang.split(" ")[0], lang.split(" ")[2],
                    sentence, head, layer, setHead, setLayer
                )}
            />
        </BaseCard>
    );
}

export default MtDemoCard;