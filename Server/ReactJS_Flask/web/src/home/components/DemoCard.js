import React from 'react'
import AsrDemoCard from './AsrDemoCard';
import MtDemoCard from './MtDemoCard';

function DemoCard(props) {
    if (props.task === "MT") {
        return <MtDemoCard/>;
    }
    else {
        return <AsrDemoCard/>;
    }
}

export default DemoCard;