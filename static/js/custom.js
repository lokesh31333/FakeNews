
// var words = {{ words | tojson }};
// var weights = {{ weights | tojson }};
// var labels = Object.keys(words);
// var data = Object.values(weights);
// var options = {
//     chart: {
//         height: 300,
//         type: "bar",
//     },
//     plotOptions: {
//         bar: {
//             horizontal: false,
//             barWidth: 60,
//             borderRadius: 2,
//             dataLabels: {
//                 position: 'top',
//                 style: {
//                     fontSize: '12px',
//                     colors: ["#304758"]
//                 },
//                 formatter: function (val) {
//                     return val.toFixed(2); // Format data labels to two decimal places
//                 }
//             },
//         },
//     },
//     dataLabels: {
//         enabled: false,
//     },
//     stroke: {
//         width: 1,
//         colors: ["#fff"]
//     },
//     series: [{
//         name: "Fake Impact",
//         data: data.map(function (weight) {
//             return weight < 0 ? null : weight; // Assign null to "Fake" bars on the left
//         })
//     }, {
//         name: "Real Impact",
//         data: data.map(function (weight) {
//             return weight > 0 ? null : weight; // Assign null to "Real" bars on the right
//         })
//     }],
//     xaxis: {
//         categories: words,
//         title: {
//             text: "Words",
//         },
//     },
//     yaxis: {
//         title: {
//             text: "Impact",
//         },
//         axisTicks: {
//             show: true,
//           },
//     },
//     fill: {
//         opacity: 1
//     },
//     legend: {
//         position: "top",
//         horizontalAlign: "left",
//         offsetX: 40,
//     },
//     colors: ["#ff9999", "#66b266"], // Light green for "Real" and light red for "Fake"
// };

// var chart = new ApexCharts(
//     document.querySelector("#apexcharts-bar"),
//     options
// );
// chart.render();
