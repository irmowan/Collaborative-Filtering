option = {
    title : {
        text: '某训练集各打分分布',
        subtext: '',
        x:'center',
        show:false
    },
    tooltip : {
        trigger: 'item',
        formatter: "{a} <br/>{b} : {c} ({d}%)"
    },
    legend: {
        orient: 'vertical',
        left: 'left',
        data: ['★','★★','★★★','★★★★','★★★★★'],
        show:false
    },

    series : [
        {
            name: '访问来源',
            type: 'pie',
            radius : '50%',
            center: ['50%', '60%'],
            data:[
                {value:4923, name:'★'},
                {value:9150, name:'★★'},
                {value:21564, name:'★★★'},
                {value:27243, name:'★★★★'},
                {value:17120, name:'★★★★★'}
            ],
            itemStyle: {
                emphasis: {
                    shadowBlur: 10,
                    shadowOffsetX: 0,
                    shadowColor: 'rgba(0, 0, 0, 0.5)',
                    label:{
                    show: true,
                    formatter: '{b} : {c} ({d}%)'
                  },
                  labelLine :{show:true}
                },
                normal:{
                  label:{
                    show: true,
                    formatter: '{b} : {c} ({d}%)'
                  },
                  labelLine :{show:true}
                }
            }
        }
        ,
    ]
};
