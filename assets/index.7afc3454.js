import{u as s,o,c as i,a as t}from"./index.842c1b90.js";const a={class:"markdown-body"},n=t("h1",null,"Boids 2D",-1),l=t("p",null,"This is a flocking agents simulation written in Unity with compute shaders as a project for the Arsiliath workshop. While some members of the workshop explored using texture-based solutions, myself and a few others explored more traditional sorting-based spatial partitioning solutions.",-1),r=t("p",null,"I built this project using Uniform Spatial Partitioning (as expertly explained in Chisolm) and learned how to implement a parallel prefix-sum operation for use in USP. The result was a simulation that could handle up to 5 million flocking agents in real time!",-1),c=t("h2",null,"Key Highlights",-1),d=t("ul",null,[t("li",null,"5 million agents interacting in complex ways in real time"),t("li",null,"Interactive simulation parameters allow for play"),t("li",null,"Customizable color gradients for more artistic expression")],-1),m=[n,l,r,c,d],w="Boids 2D",x="2021-01-12T00:00:00.000Z",k="2021-01-15T00:00:00.000Z",b="An agents simulation with flocking behaviors.",T=1,y=["SHADER","SIMULATION"],A=[],B={__name:"index",setup(h,{expose:e}){return e({frontmatter:{name:"Boids 2D",start_date:"2021-01-12T00:00:00.000Z",release_date:"2021-01-15T00:00:00.000Z",short_description:"An agents simulation with flocking behaviors.",carousel_aspect:1,categories:["SHADER","SIMULATION"],meta:[]}}),s({meta:[]}),(_,g)=>(o(),i("div",a,m))}};export{T as carousel_aspect,y as categories,B as default,A as meta,w as name,k as release_date,b as short_description,x as start_date};