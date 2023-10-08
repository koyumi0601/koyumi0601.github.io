function fnScroll(leftFixedCls , rightFixedCls){//고정 시킬 좌 우 클래스 네임

    var lfH = $('.' + leftFixedCls).outerHeight() , boxH = $('.box_benefit').height() , rgH = $('.' + rightFixedCls).outerHeight() , headerH = 128;
    var boxTop = $('.box_benefit').offset().top + 49;//스크롤 event 시작지점
    var rgTop = $('.' + rightFixedCls).offset().top;//right_column 스크롤 event 시작시점
    var scrollBtm =  boxH - lfH + rgH + headerH;//스크롤 이벤트 중간 분기점
    var scrollTop = $(window).scrollTop();
    if(boxH > lfH){ //스크롤 이벤트 시작 조건 우측 콘텐트 길이가 길때 이벤트 발생
        //스크롤이 진행된 상황에서 새로고침 시 class 조건
        if(scrollTop > boxTop){ //left_fixed 조건
            $('.' + leftFixedCls).addClass('fixed');
        } else {
            $('.' + leftFixedCls).removeClass('fixed');
        }

        if(scrollTop > rgTop){//right_fixed 조건
            $('.' + rightFixedCls).addClass('fixed');
        } else {
            $('.' + rightFixedCls).removeClass('fixed');
        }

//        if(scrollTop > scrollBtm){
//            $('.box_benefit').addClass('act');
//        } else {
//            $('.box_benefit').removeClass('act');
//        }

        $(window).scroll(function(){
            var scrollTop = $(window).scrollTop();
            var boxTop = $('.box_benefit').offset().top - 49;//스크롤 event 시작지점
            var scrollBtm =  boxH - lfH + rgH + headerH;//스크롤 이벤트 중간 분기점
            //console.log(scrollBtm , boxH , lfH , $(window).scrollTop())
            if(scrollTop > boxTop){ //left_fixed 조건
                $('.' + leftFixedCls).addClass('fixed');
            } else {
                $('.' + leftFixedCls).removeClass('fixed');
            }

            if(scrollTop > rgTop){//right_fixed 조건
                $('.' + rightFixedCls).addClass('fixed');
            } else {
                $('.' + rightFixedCls).removeClass('fixed');
            }

//            if(scrollTop >= scrollBtm){
//                $('.box_benefit').addClass('act');
//            } else {
//                $('.box_benefit').removeClass('act');
//            }
        });
    }
}
// 220126 서브메인 고정 바 Q10048
var myBen = {
    init : function() {
        _this = this
        $('.sub_main_login .con1 .right').removeClass('scroll-able fix').removeAttr('style');
        var leftH = $('.sub_main_login .con1 .left').outerHeight() + 100;
        var rightH = $('.sub_main_login .con1 .right').outerHeight();
        if(leftH > rightH){
            _this.scroll();
        } else {
            $(window).off('scroll');
        }
    },
    update : function(){
        var headerConH = $('.header').outerHeight() + $('.header_sub').outerHeight(); // header height
        $('.sub_main_login .con1 .right').addClass('scroll-able fix').css({'top': headerConH + 48 , 'bottom': '' , 'transition-property':'top' , 'transition-duration':'1s' , 'transition-delay':'.4s'});
        _this.scroll();
    },
    scroll : function(){
        var PossibleH = $('.sub_main_login .con1 .left').outerHeight() - $('.sub_main_login .con1 .right').outerHeight(); //스크롤 가능한 영역
        $(window).on('scroll', function () {
            var scrollTop = $(this).scrollTop(); // 스크롤 top 반환
            var scrollH = PossibleH - scrollTop; //스크롤 위치 계산
            var headerConH = $('.header').outerHeight() + $('.header_sub').outerHeight(); // header height
            if (scrollTop > 0 && PossibleH > 0) {
                $('.sub_main_login .con1 .right').addClass('scroll-able');// position fixed
                if (scrollH < 0) {
                    $('.sub_main_login .con1 .right').css({'top': 'auto', 'bottom' : 24}).removeClass('fix');
                } else {
                    $('.sub_main_login .con1 .right').addClass('scroll-able fix').css({'top': headerConH + 48 , 'bottom': ''});
                } 
            } else { 
                    $('.sub_main_login .con1 .right').removeClass('scroll-able fix').removeAttr('style');
            }
        });
    }
}

$(window).load(function(){
    $(document).ready(function(){
        $(document).on('click', '.accod_btn', function(){
            var _this = $(this) , title = _this.text();
            _this.attr('title' , title + ' 리스트 닫기').toggleClass('on').parents('.items').children('.accod_slide').slideToggle();
            if(_this.is('.on') === true){
                _this.attr('title' , title + ' 리스트 열기')
            }
        });
    });
    if ($('.sub_main_login').length) { myBen.init(); }

    $(document).on('click', '.sub_main_login #moreBtn' , function(){ //더보기 버튼 클릭 시
        myBen.update();
    });

});

$(document).ready(function($){
    $(".scroll_move").click(function(event){
        event.preventDefault();
        if( $(this.hash).closest(".accodWrap").hasClass("on") ) {
            $('html,body').animate({scrollTop:$(this.hash).closest(".accodWrap").offset().top - 96}, 500);
        } else {
            $(this.hash).closest(".accodWrap").addClass("on").find(".accodSlide").slideDown(300);
            $('html,body').animate({scrollTop:$(this.hash).closest(".accodWrap").offset().top - 96}, 500);
        }
    })
    // 2022-10-18 GGCP31 이벤트 배경 색상변경하기 (임시적으로 변경) 이벤트코드 5Q2320
    if (location.href.indexOf("5Q2320") > 0) {
      $(".title_box > .h3_b_lt").addClass("fc_wht");
      $(".title_box > .h3_b_lt + p").removeClass("fc_m_a64").addClass("fc_wht").css("opacity", "64%");
    };

    // 2023-05-10 Q10962 - [이벤트]Apple Pay 5월 결제 혜택 이벤트페이지 신규 제작의 건 이벤트 코드 1NH405 / 2023-05-26 Q11139 - 6월 이벤트 코드 VG7417
    if (location.href.indexOf("1NH405") > 0 || location.href.indexOf("VG7417") > 0) {
        $(window).scroll(function(){
            var winTop = $(window).scrollTop();
            var headHeight = 60; // tabHeight
            var sTop = winTop + headHeight;
            var uiTabTop = $('.tab_wrap').offset().top - 96;
            if ( winTop > uiTabTop ){
                $('.sub_tab').addClass('fixed');
            } else {
                $('.sub_tab').removeClass('fixed');
            }
    
            $('.cont').each(function(index){
                var $menu = $(this).closest('.container');
                var liTop = $menu.find('.cont').eq(index).offset().top;
                var panelHei = $menu.find('.cont').eq(index).outerHeight() / 7;
                if(liTop - panelHei <= sTop + headHeight ){
                    $('.tab_btn_list>li').removeClass('on').find('a').removeAttr('title');
                    $('.tab_btn_list>li').eq(index).addClass('on').find('a').attr('title','선택됨')
                } 
            })
            var onIndex = $('.tab_btn_list>li.on').index();
            tabScroll(onIndex+1);
        })
        function tabScroll(idx){
            $('.tab_btn_list').each(function(){
                if ($(this).find('li').hasClass('on')){
                    if(idx){
                        var $item = $(this).find('li:nth-child('+idx+')');
                        $(this).find('li').removeClass('on').find('a').removeAttr('title');
                        $item.addClass('on').find('a').attr('title','선택됨');
                    }
                }
            })
        }
    
        $('.tab_btn_list').on('click', '.tab_btn',function(e){
            e.stopPropagation();
            var _thisTab = $(this).data('tab-scroll');
            var _thisTabParent = $(this).parent();
            _thisTabParent.addClass('on').siblings().removeClass('on');
            var _tabTop = $('.'+_thisTab).offset().top - 156;
            $('html,body').stop().animate({scrollTop:_tabTop},500);
            var onIndex = $('.tab_btn_list>li.on').index();
            tabScroll(onIndex+1);
        })

        // cont01 : 편의점, cont02 : 백화점/쇼핑, cont03 : 마트/슈퍼, cont04 : 커피, cont05 : 제과디저트, cont06 : 외식, cont07 : 생활가전, cont08 : 호텔리조트, cont09 : 주유충전, cont10 : 영화도서, cont11 : 레저여행, cont12 : 온라인
        var _fromVal = getParamsFromUrl(window.location.href);
        var _fromRegex  = /[^0-9]/g; 
        var _fromResult = _fromVal.from === undefined ? '' : _fromVal.from.replace(_fromRegex, "");

        if (location.href.indexOf("1NH405") > 0 ){
            switch(_fromResult){
                case '01':
                    $('.cont02, .cont12').remove();
                    $('.tab_btn_02, .tab_btn_12').parent().remove();
                    break;
                case '02':
                    $('.cont01, .cont02, .cont03, .cont07, .cont12').remove();
                    $('.tab_btn_01, .tab_btn_02, .tab_btn_03, .tab_btn_07, .tab_btn_12').parent().remove();
                    break;
                case '03':
                    $('.cont09').remove();
                    $('.tab_btn_09').parent().remove();
                    break;
                default:
                    console.log('일반진입')
            }
        }
        if (location.href.indexOf("VG7417") > 0 ){
            switch(_fromResult){
                case '01':
                    break;
                case '02':
                    break;
                case '03':
                    break;
                case '04':
                    $('.cont01, .cont02, .cont03, .cont07, .cont12').remove();
                    $('.tab_btn_01, .tab_btn_02, .tab_btn_03, .tab_btn_07, .tab_btn_12').parent().remove();
                    break;
                case '05':
                    $('.cont09').remove();
                    $('.tab_btn_09').parent().remove();
                    break;
                default:
                    console.log('일반진입')
            }
        }
    };
});



