import React, { useState } from 'react'
import { makeStyles} from '@material-ui/core/styles';
import {Stepper, Step, StepLabel, StepContent, Button,  Typography, LinearProgress} from '@material-ui/core'
import {fetchModelInferenceResult} from '../inference';

const useStyles = makeStyles((theme) => ({
    demoSteps: {
        backgroundColor: "#F5F5F5",
        borderRadius: 15,
        margin: 15
    },
    actionsContainer: {
        marginBottom: theme.spacing(2),
        float: "right"
    },
    button: {
        marginRight: theme.spacing(1),
        marginBottom: 20,
        marginTop: 20,
    },
}));

function NextButton(props) {
    const classes = useStyles();

    return (
        props.activeStep === 0 ? (<div className={classes.actionsContainer}>
            <div>
                <Button
                    variant="contained"
                    color="primary"
                    onClick={() => props.handleNext()}
                    className={classes.button}
                >
                    Next
                </Button>
            </div>
        </div>) : null
    );
}

function ResetButton(props) {
    const classes = useStyles();
    return (props.inferenceFinish && (
        <div style={{float: "none", textAlign: "right"}}>
            <Button
                onClick={props.handleReset}
                style={{width: 150}}
                className={classes.button} 
                olor="primary"
                variant="contained"
            >
                Try again
            </Button>
        </div>       
    ));
}

function DemoStepContent(props) {
    if (props.step === 0) {
        return props.inputContent(props.reset);
    }
    else if (props.step === 1) {
        return <LinearProgress color="primary"/>;
    }
    else if (props.step === 2) {
        return props.resultContent(props.inferenceFinish, props.inferenceResult, props.reInferenceModel);
    }
    else {
        return "Error"
    }
}

function DemoSteps(props) {
    /* DemoSteps contains 3 steps.
    1. Show input content
    2. Model inference
    3. Show inference result
    */
    const classes = useStyles();
    const [activeStep, setActiveStep] = React.useState(0);
    const [inferenceFinish, setInferenceFinish] = useState(false);
    const [inferenceResult, setInferenceResult] = useState({});
    var inferenced = false;

    const handleNext = () => {
        if (!inferenced) {
            inferenced = true;
            fetchModelInferenceResult(
                props.data(), props.serverIp(), handleModelInferenceFinish
            );
        }
        setActiveStep((prevActiveStep) => prevActiveStep + 1);
    };

    const handleReset = () => {
        setActiveStep(0);
        setInferenceFinish(false);
        inferenced = false;
    };

    const handleModelInferenceFinish = (inferenceResult) => {
        setInferenceFinish(true);
        setActiveStep((prevActiveStep) => prevActiveStep < 2 ? prevActiveStep + 1 : 2);
        setInferenceResult(inferenceResult);
    };

    const reInferenceModel = () => {
        setInferenceFinish(false);
        fetchModelInferenceResult(
            props.data(), props.serverIp(), handleModelInferenceFinish
        );
    }

    return (
        <Stepper activeStep={activeStep} orientation="vertical" className={classes.demoSteps}>
            {
                props.steps.map((label, step) => (
                    <Step key={label}>
                        <StepLabel>
                            <Typography align="left">{label}</Typography>
                        </StepLabel>
                        <StepContent>
                            <ResetButton
                                inferenceFinish={inferenceFinish}
                                handleReset={handleReset}
                            />
                            
                            <DemoStepContent
                                step={step}
                                reset={handleReset}
                                inputContent={props.inputContent}
                                resultContent={props.resultContent}
                                inferenceFinish={inferenceFinish}
                                inferenceResult={inferenceResult}
                                reInferenceModel={reInferenceModel}
                            />

                            <NextButton
                                activeStep={activeStep}
                                handleNext={handleNext}
                            />
                        </StepContent>
                    </Step>
                ))
            }
        </Stepper>
    );
}

export default DemoSteps;