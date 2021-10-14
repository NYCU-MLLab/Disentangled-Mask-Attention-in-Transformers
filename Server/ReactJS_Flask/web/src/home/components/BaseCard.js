import React from 'react'
import { makeStyles } from '@material-ui/core/styles';
import {Typography, Card, CardContent, Divider} from '@material-ui/core'

const useStyles = makeStyles((theme) => ({
    baseCard: {
        margin: "10px",
        marginRight: 5,
        marginBottom: 0,
        backgroundColor: "#F5F5F5",
        variant: "outlined",
        zIndex: 1
    },
}));

function BaseCard(props) {
    const classes = useStyles();

    return (    
        <Card className={classes.baseCard} style={props.style}>
            <CardContent style={{paddingBottom: 0, backgroundColor: "#EEEEEE"}}>
                <Typography variant="h6" style={{marginBottom: 10}}><b>{props.title}</b></Typography>
                <Divider />
            </CardContent>
            <CardContent style={{verticalAlign: "middle"}}>
                {props.children}
            </CardContent>
        </Card>
    );
}

export default BaseCard;