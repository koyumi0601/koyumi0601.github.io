(function($, window, document, undefined){
    
    /**
     * 일정 시간동안 한 페이지에 머물게 되면 세션 종료 경고 레이어가 뜨게 되고
     * 레이어 내부의 버튼에 따라 세션 연장 및 세션 종료가 된다.
     * 
     * pc - htmlHead.jsp
     *            
     *      $.hcSessionTime();
     *            
     *            
     * mobile - htmlHead.jsp
     *            
     *      var config = {
     *          deviceType : "MOBILE",   // 레이어 구분
     *          fromApp : "false"        // 접근 경로 구분
     *      }
     *      
     *      $.hcSessionTime(config); 
     */
    
    var hcSession = function(options) {
        this.options = {};
        this.options = options;
    };
    
    hcSession.prototype = {
        defaults : 
        {
            sessionTime : 600,      // 세션 만료 시간 600
            currentTime : 600,      // 세션 만료 시간 600
            alertTime : 180,        // 세션 만료 안내 표시 시간  
            tid : null ,            // setInterval object ID
            deviceType : "PC",      // Device Type
            fromApp : false,         // app에서 호출여부
            isLogined   : true      // 로그인사용자 여부
        },
        /**네
         * hcSession 초기 셋팅. 세션 만료 시간 체크 시작.
         */
        init:function() {
            var self = this;
            self.config = $.extend({}, self.defaults, self.options);
            console.log("init self.config.fromApp:"+self.config.fromApp);
            if(!self.config.fromApp){ 
                self.config.currentTime = self.config.sessionTime;
                self.config.tid = setInterval($.proxy(self.showTime, self), 1000);
            }
        },
        
        /**
         * Device에 따라 세션 만료 레이어를 띄우고 위치를 선정한다.
         */
        drawAlertLayer : function() {
            //console.log("self.currentTime:"+self.currentTime+", self.alertTime:"+self.alertTime); 
            var self = this;
            var data = '';
            if (!self.config.isLogined) {
                console.log("not login timeout pop"); 
                modal.open('popup_layer_timeout', '');
            }
            else{
                popup.open('popDisSS', '');
            }
            
            //console.log("drawAlertLayer $.proxy(self.layerEventHandle, self)");
            //setTimeout($.proxy(self.layerEventHandle, self), 500);
            
            //LAYER();
        },
        
        /**
         * 세션 레이어 내부의 버튼들에 대한 이벤트를 정의한다.
         */
        /*
        layerEventHandle : function(){          
            var self = this;
            
            var $timegoes = $('#timegoes');
            var $timeout = $('#timeout');
                        
            $timeout.off("click.session");
            $timegoes.off("click.session");

            $timeout.on("click.session", function(e){ self.close(e); });
            $timegoes.on("click.session", function(e){ self.extend(e); });
        },
        */
        /**
         * 레이어에 시간을 표시하고 세션 만료시간이 되었을 때 세션 종료 함수를 호출한다.
         */
        showTime : function() {
            var _this = this;
            var self = _this.config;
            
            //console.log(self.capturedMin = Math.floor(self.sessionTime / 60),"분 ",self.sessionTime % 60,"초");            
            var currentMin = Math.floor(self.currentTime / 60);
            var currentSec = self.currentTime % 60;
            if(currentMin < 0){
                currentMin = 0;
            }
            if(currentSec < 0){
                currentSec = 0;
            }
            $("#topMinSec").html(((currentMin < 10 )?"":"") + currentMin+"분 "+((currentSec < 10 )?"0":"") + currentSec+"초");
            $("#capturedMinSec").html(((currentMin < 10 )?"":"") + currentMin+"분 "+((currentSec < 10 )?"0":"") + currentSec+"초 후");
            
            if(self.currentTime == self.alertTime) {
                _this.drawAlertLayer();
            } else if(self.currentTime <= 0) {  //세션 종료
                clearInterval(self.tid);
                if (self.alertTime > 0) {
                    _this.close();
                    //$('#timeout_layout').remove();
                }
            }
            self.currentTime--;
        },
        
        /**
         * 레이어 화면의 닫기 이벤트를 호출한다.
         * @param ev
         */
        cancel : function(ev) {      // 레이어 닫기
            if(ev != null && typeof ev != "undefined") {
                ev.preventDefault();
                ev.stopPropagation();
            }
            setTimeout(function(){
                popup.close('popDisSS', 'btnOpenPopSs');//현재 팝업 닫기
            },300);
        },
        
        /**
         * 레이어 화면의 세션종료 이벤트를 호출한다
         * @param ev
         */
        extend : function(ev) {         // 시간 연장
            console.log("gose extend:"+ev);
            if(ev != null && typeof ev != "undefined") {
                ev.preventDefault();
                ev.stopPropagation();
            }
            var self = this;
            $.hcAjax({
                category:"data",
                url:"/cpm/mb/apiCPMMB0101_05.hc",
                success:function(){
                    self.cancel(ev);
                    self.config.currentTime = self.config.sessionTime;
                }
            });
            //$('.dimmed').remove();
			popup.close('popDisSS', 'btnOpenPopSs');//현재 팝업 닫기
        },
        reset : function(ev) {         // 시간 연장
            var self = this;
            self.config.currentTime = self.config.sessionTime;
        },
        close : function(ev){
            //console.log("close ev:"+ev);
            if(ev != null && typeof ev != "undefined") {  // 세션 끊기 
                ev.preventDefault();
                ev.stopPropagation();
            }
            location.href="/cpm/mb/CPMMB1001_02.hc";
            //asis location.href="/common/error/commonLogout.hc";
        }
    };
    
    hcSession.defaults = hcSession.prototype.defaults;

    $.hcSessionTime = function(options) {
        var sessionTime = new hcSession(options);
        sessionTime.init();
        return sessionTime;
    };
})(jQuery, window);
