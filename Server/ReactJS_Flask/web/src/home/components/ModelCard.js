import React from 'react'
import BaseCard from './BaseCard'

function ModelCard(props) {
    return (
        <BaseCard title={"Disentangled mask attention"} style={{height: 525, overflow: "auto"}}>
            <img
                src="Disentangled-mask-attention.png"
                style={{height: 400, overflow: "auto", verticalAlign: "middle"}}
                alt="Model"
            />
        </BaseCard>
    );
}

export default ModelCard;