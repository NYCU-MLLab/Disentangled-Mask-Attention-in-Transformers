import React from 'react'
import {InputLabel, Select, Typography, Card, Grid, LinearProgress, Divider, FormControl, MenuItem} from '@material-ui/core'

function AttentionTable(props) {
    const title = ["Encoder raw attention", "Encoder attentoin mask", "Encoder attention weights",
                   "Decoder raw attention", "Decoder attentoin mask", "Decoder attention weights",
                   "Decoder-Encoder raw attention", "Decoder-Encoder attentoin mask",
                   "Decoder-Encoder attention weights"];
    const key = ["enc_attn", "enc_mask", "enc_mask_attn",
                 "dec_attn", "dec_mask", "dec_mask_attn",
                 "dec_enc_attn", "dec_enc_mask", "dec_enc_mask_attn"];

    return (
        <div>
            <div style={{textAlign: "center"}}>
                <div style={{paddingLeft: 150}}>
                    <FormControl style={{margin: 10, marginBottom: 30, minWidth: 100}}>
                        <InputLabel>Layer</InputLabel>
                        <Select value={props.layer} onChange={(event) => {
                            props.setLayer(event.target.value);
                            props.reInferenceModel();
                        }}>
                            {[...Array(6).keys()].map(i => (
                                <MenuItem value={i+1}>{i+1}</MenuItem>
                            ))}
                        </Select>
                        
                    </FormControl>
                    <FormControl style={{margin: 10, marginBottom: 30, minWidth: 100}}>
                        <InputLabel>Head</InputLabel>
                        <Select value={props.head} onChange={(event) => {
                                props.setHead(event.target.value);
                                props.reInferenceModel();
                            }}>
                            {[...Array(4).keys()].map(i => (
                                <MenuItem value={i+1}>{i+1}</MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </div>
                {
                    !props.inferenceFinish ? <LinearProgress style={{margin: 25}}/> : null
                }
            </div>
            <Grid container spacing={1}>
                {
                    title.map((title, index) => (
                        <Grid item xs={4}>
                            <Card>
                                <Typography>{title}</Typography>
                                <Divider/>
                                <img
                                    src={"data:image/png;base64, "+props.inferenceResult[key[index]]}
                                    width="100%"
                                    alt="Attention map"
                                />
                            </Card>
                        </Grid>
                    ))
                }
            </Grid>
        </div>
    );
}

export default AttentionTable