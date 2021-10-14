import React, { useEffect, useState } from 'react'
import {makeStyles } from '@material-ui/core/styles';
import {Typography} from '@material-ui/core'
import BaseCard from './BaseCard'

const useStyles = makeStyles((theme) => ({
    abstractCard: {
        margin: "10px",
        verticalAlign: "middle",
        padding: 0,
        backgroundColor: "#F5F5F5",
        variant: "outlined",
        overflow: "auto",
        textAlign: "left"
    },
}));

function AbstractCard(props) {
    const classes = useStyles();
    const [abstract, setAbastact] = useState("");

    useEffect(() => {
        var abstractFile = new XMLHttpRequest();
        abstractFile.open("GET", "./abstract.txt", false);
        abstractFile.onreadystatechange = () => {
            if (abstractFile.readyState === 4) {
                if (abstractFile.status === 200 || abstractFile.status === 0) {
                    setAbastact(abstractFile.responseText)
                }
            }
        };
        abstractFile.send(null);
    });

    return (
        <BaseCard title={"Abastact"} style={{height: 525, overflow: "auto"}}>
            <Typography className={classes.abstractCard}>
                {abstract}
            </Typography>
        </BaseCard>
    );
}

export default AbstractCard;