import React, { useState } from "react";
import HomeNavBar from "./NavBar";
import {Grid} from '@material-ui/core'
import AbstractCard from "./components/AbstractCard"
import ModelCard from "./components/ModelCard"
import DemoCard from "./components/DemoCard"

function HomePage() {
    const [task, setTask] = useState("MT");

    return (
        <div>
            <div style={{position: "fixed", width: "100%", zIndex: 2, margin: 0}}>
                <HomeNavBar task={task} onTaskChange={(event) => {setTask(event.target.value)}}/>
            </div>
            <div style={{marginTop: 85, backgroundColor: "#BDBDBD", verticalAlign: "top"}}>
                <Grid container spacing={0}>
                        <Grid item xs={5}>
                            <ModelCard/>
                    </Grid>
                    <Grid item xs={7}>
                        <AbstractCard/>
                    </Grid>
                </Grid>
                <DemoCard task={task} style={{height: 0}}/>
            </div>
        </div>
    );
}

export default HomePage