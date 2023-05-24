// function loadJSON(filename) {
//     var symbol = filename.substring(0, filename.indexOf('_signal'));
//     const path = '../static/file/';
//     let filepath = path + filename;
//     console.log(symbol);
//     return fetch(filepath)
//       .then(response => response.json())
//       .then(data => {
//         return data; // Return the loaded JSON data
//       })
//       .catch(error => {
//         console.error('Error:', error);
//       });
// }



chart = {

    
    // build: function () {
    //     let object = this;

    //     console.log("Building chart...");
    //     object.getData(function(data) {
    //         object.buildHighchart(data);
    //         object.buildAnalysisHoloDemo(data);  
    //     })
    // },
    
    // getData: function (callback) {
    //     fetch('../static/file/AAPL_signal.json')
    //         .then(response => response.json())
    //         .then(data => {
    //             if (callback) {
    //                 callback(data);
    //             }
    //         })
    //         .catch(error => console.error(error));
    // },
    getData: function(filename) {
        const path = '../static/file/';
        let filepath = path + filename;
        return fetch(filepath)
          .then(response => response.json())
          .then(data => {
            return data; // Return the loaded JSON data
        })
          .catch(error => {
            console.error('Error:', error);
        });
    },
    
      build: function() {
        let object = this;
    
        console.log("Building chart...");
        if (object.currentData) {
          object.buildHighchart(object.currentData); // Use the current data if available
          object.buildAnalysisHoloDemo(object.currentData);
        } else {
          console.log("No data available.");
        }
    },


    buildHighchart: function (data) {
        let object = this;
        // split the data set into ohlc and volume
        var ohlc = [],
            volume = [],
            formatData = [],
            dataLength = data.length;

        for (var i = 0; i < dataLength; i += 1) {
            ohlc.push([
                new Date(data[i]["date"].split("/").reverse().join("/")).getTime(),   // the date
                data[i]["1. open"],                         // open
                data[i]["2. high"],                         // high
                data[i]["3. low"],                          // low
                data[i]["4. close"]                         // close
            ]);

            volume.push([
                new Date(data[i]["date"].split("/").reverse().join("/")).getTime(),        // the date
                data[i]["6. volume"],                          // the volume
            ]);

            formatData.push([
                new Date(data[i]["date"].split("/").reverse().join("/")).getTime(),   // the date
                Number(data[i]["1. open"]),                         // open
                Number(data[i]["2. high"]),                         // high
                Number(data[i]["3. low"]),                          // low
                Number(data[i]["4. close"]),                        // close
                data[i]["6. volume"],                       // the volume
            ])
        }

        // create the chart
        Highcharts.stockChart('candlechart', {
            chart: {
                height: 600
            },
            title: {
                text: 'AAPL' + ' Historical'
            },
            subtitle: {
                text: 'All indicators'
            },
            accessibility: {
                series: {
                    descriptionFormat: '{seriesDescription}.'
                },
                description: 'Use the dropdown menus above to display different indicator series on the chart.',
                screenReaderSection: {
                    beforeChartFormat: '<{headingTagName}>{chartTitle}</{headingTagName}><div>{typeDescription}</div><div>{chartSubtitle}</div><div>{chartLongdesc}</div>'
                }
            },
            legend: {
                enabled: true
            },
            rangeSelector: {
                selected: 2
            },
            yAxis: [{
                height: '60%'
            }, {
                top: '60%',
                height: '20%'
            }, {
                top: '80%',
                height: '20%'
            }],
            plotOptions: {
                series: {
                    showInLegend: true,
                    accessibility: {
                        exposeAsGroupOnly: true
                    }
                }
            },
            series: [{
                type: 'candlestick',
                id: 'aapl',
                name: 'AAPL',
                data: formatData,
            }, {
                type: 'column',
                id: 'volume',
                name: 'Volume',
                data: volume,
                yAxis: 1
            }, {
                type: 'pc',
                id: 'overlay',
                linkedTo: 'aapl',
                yAxis: 0
            }, {
                type: 'macd',
                id: 'oscillator',
                linkedTo: 'aapl',
                yAxis: 2
            }]
        }, function (chart) {
            document.getElementById('overlays').addEventListener('change', function (e) {
                var series = chart.get('overlay');

                if (series) {
                    series.remove(false);
                    chart.addSeries({
                        type: e.target.value,
                        linkedTo: 'aapl',
                        id: 'overlay'
                    });
                }
            });

            document.getElementById('oscillators').addEventListener('change', function (e) {
                var series = chart.get('oscillator');

                if (series) {
                    series.remove(false);
                    chart.addSeries({
                        type: e.target.value,
                        linkedTo: 'aapl',
                        id: 'oscillator',
                        yAxis: 2
                    });
                }
            });
        });

    },

    formatDataSet: function(dataset) {
        let json = {
            "Total": {},
            "BUY": {},
            "NEU": {},
            "SELL": {},
        }
        dataset.forEach(data => {
            let year = data.date.split("/")[2];

            json["Total"][`${year}`] = `${data['BUY'] + data['SELL'] + data['NEU']}`;
            json["BUY"][`${year}`] = `${data['BUY']}`;
            json["NEU"][`${year}`] = `${data['NEU']}`;
            json["SELL"][`${year}`] = `${data['SELL']}`;
        });

        return json
    },

    buildAnalysisHoloDemo: function(response) {
        let object = this;

        const startYear = 1999,
        endYear = 2023,
        btn = document.getElementById('play-pause-button'),
        input = document.getElementById('play-range'),
        nbr = 6;
    
        let chart;
        let dataset = object.formatDataSet(response);
        //console.log("test output", (dataset))
        
        function getData(year) {
            const output = Object.entries(dataset).map(country => {
            const [countryName, countryData] = country;
                return [countryName, Number(countryData[year])];
            });
            return [output[0], output.slice(1, nbr)];
        }

       
        function getSubtitle() {
            const totalNumber = getData(input.value)[0][1].toFixed(2);
            return `<span style="font-size: 80px">${input.value}</span>
                <br>
                <span style="font-size: 22px">
                    Total: <b> ${totalNumber}</b> TWh
                </span>`;
        }
        
    
        
        chart = Highcharts.chart('container', {
            title: {
                text: 'Technical Analysis',
                align: 'center'
            },
            subtitle: {
                useHTML: true,
                text: getSubtitle(),
                floating: true,
                verticalAlign: 'middle',
                y: 30
            },
    
            legend: {
                enabled: false
            },
    
            tooltip: {
                valueDecimals: 2,
                valueSuffix: ' TWh'
            },
    
            plotOptions: {
                series: {
                    borderWidth: 0,
                    colorByPoint: true,
                    type: 'pie',
                    size: '100%',
                    innerSize: '80%',
                    dataLabels: {
                        enabled: true,
                        crop: false,
                        distance: '-10%',
                        style: {
                            fontWeight: 'bold',
                            fontSize: '16px'
                        },
                        connectorWidth: 0
                    }
                }
            },
            colors: ['#FCE700', '#F8C4B4', '#f6e1ea', '#B8E8FC', '#BCE29E'],
            series: [
                {
                    type: 'pie',
                    name: startYear,
                    data: getData(startYear)[1]
                }
            ]
        });
                
        /*
        * Pause the timeline, either when the range is ended, or when clicking the pause button.
        * Pausing stops the timer and resets the button to play mode.
        */
        function pause(button) {
            button.title = 'play';
            button.className = 'fa fa-play';
            clearTimeout(chart.sequenceTimer);
            chart.sequenceTimer = undefined;
        }
        
        /*
        * Update the chart. This happens either on updating (moving) the range input,
        * or from a timer when the timeline is playing.
        */
        function update(increment) {
            if (increment) {
                input.value = parseInt(input.value, 10) + increment;
            }
            if (input.value >= endYear) {
                // Auto-pause
                pause(btn);
            }
        
            chart.update(
                {
                    subtitle: {
                        text: getSubtitle()
                    }
                },
                false,
                false,
                false
            );
        
            chart.series[0].update({
                name: input.value,
                data: getData(input.value)[1]
            });
        }
        
        /*
        * Play the timeline.
        */
        function play(button) {
            button.title = 'pause';
            button.className = 'fa fa-pause';
            chart.sequenceTimer = setInterval(function () {
                update(1);
            }, 100);
        }
        
        btn.addEventListener('click', function () {
            if (chart.sequenceTimer) {
                pause(this);
            } else {
                play(this);
            }
        });
        /*
        * Trigger the update on the range bar click.
        */
        input.addEventListener('input', function () {
            update();
        });  
    },
};