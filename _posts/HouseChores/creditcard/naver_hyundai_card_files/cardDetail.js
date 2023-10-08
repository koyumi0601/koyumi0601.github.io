/* cardDetail PC */
var idx = 0; // 2023-04-03 GGU282 - 초기값 지정
var detailSwiper; // 2023-04-03 GGU282 - 전역 변수로 선언
$(document).on('click' , '.card_design_wrap ul > li > a' , function(){
    idx = $(this).parent('li').index();//클릭한 부모인자 idx
});

var detail = {
    open : function(id , btn){
        // 2023-04-03 GGU282 - 추가 s
        if(detailSwiper) {
            detailSwiper.destroy();
            detailSwiper = null;
        }
        // 2023-04-03 GGU282 - 추가 e

        hearderTitle = $('#'+id).find('.swiper-slide').eq(idx).attr('data-header-title');
        hearderCon = $('#'+id).find('.swiper-slide').eq(idx).attr('data-header-con');
        $('#'+id).find('.layer_head h3').text(hearderTitle);//클릭시 초기 header title 셋팅
        $('#'+id).find('.layer_head p').text(hearderCon);//클릭시 초기 header title 셋팅
        // $('.caution_list').html(html);//유의사항을 공통으로 사용하지 않을경우 해당 부분 주석처리
        popup.open(id , btn);//popup open
        slideIndex = $('#'+id).find('.swiper-slide').index();

        if($('#popCardSelect .swiper-slide').length !== 1){

            // 2023-04-03 GGU282 - detailSwiper 전역 변수로 선언
            detailSwiper = new Swiper('#' + id + ' .swiper-container', {//q10011 0721 복수개의 스와이퍼 팝업이 있는 케이스로 해당 아이디로 실행
                loop: true,
                autoHeight : true,
                navigation : {
                    prevEl : '#' + id + ' .swiper-button-prev',
                    nextEl : '#' + id + ' .swiper-button-next'
                },
                pagination : {
                    el : '.modal_card_view .swiper-pagination',
                    clickable :true,
                }

            });
            detailSwiper.on('activeIndexChange' , function(){
                hearderTitle = $('#'+id).find('.swiper-slide').eq(this.activeIndex).attr('data-header-title');
                hearderCon = $('#'+id).find('.swiper-slide').eq(this.activeIndex).attr('data-header-con');
                $('#'+id).find('.layer_head h3').text(hearderTitle);
                $('#'+id).find('.layer_head p').text(hearderCon);

                commonUi.scrbarUpdate.init($(this).closest('.modal_pop').attr('id'), 'static'); // 컨텐츠 크기 커질 경우 스크롤 업데이트 추가 Q10048
            });
            setTimeout(function(){detailSwiper.slideTo(idx + 1 , 500 , false);} , 200)

            /* 2023-04-03 GGU282 - 삭제, dim 클릭으로 닫은 경우 실행되지 않아서 open 시 처리하도록 수정
            $(document).on('click' , '.layer_close a' , function(){
                detailSwiper.destroy();
            }); */
        };

    },
    scrollIsu: function() {
        var $window = $(window);
        var $detail = $('.card_detail_view');
        var _benefit = $detail.find('.card_benefit');

        if (_benefit.length){
            $window.on('scroll', function(){
                var st = $window.scrollTop();
                var headerH = $('header').height();

                if(st >= (_benefit.offset().top - headerH)) {
                    _benefit.addClass('on');
                } else {
                    _benefit.removeClass('on');
                }
            });
        }
    }
}

var remote = {
    eventCheck : 0,
    init: function () {
        if(this.eventCheck === 0){
            var remoteH,
                remoteHbig,
                remoteImgLength = $('.card_remote img').length,
                imgLoadedNum = 0;
            $('.card_remote img').on('load', function(){ // 이미지가 로드 되었는 지 확인 (시점에 따라 높이값 체크 오류 있음)
                imgLoadedNum ++;
                if( remoteImgLength == imgLoadedNum){
                    remoteH = $('.card_remote').outerHeight(); //큰 리모콘 높이
                    $('.card_remote').addClass('off'); // 작은리모콘으로 셋팅
                    remoteHbig = $('.card_remote.off').outerHeight(); //작은 리모콘 높이

                    setTimeout(function () { //높이 체크 시 안보이도록 천천히 노출
                        $('.card_remote').css('opacity', '1');
                    }, 100);
                }
            })

            this.eventCheck ++;
            console.log("card_remote" + this.eventCheck);
            //hover
            $('.card_remote').on('mouseenter focusin click', function () {
                $(this).css('height', remoteH).removeClass('off')
            });
            $('.card_remote').on('mouseleave', function () {
                $(this).css('height', remoteHbig).addClass('off')
            });
        }
    }
}

/* s : 2022-05-25 GGCN45 : window.onload 내부 FN 리팩토링 */
var cardBasicInteractionFn = {
    initSnackBarFn : false,
    initArrowScrollFn : false,
    initFloatingBannerMotionFn : false,
    init: function() {
        var _this = this;
        detail.scrollIsu();
        if($(".member_fee") && $('.cont_wrap .apply_btn') && !this.initSnackBarFn){this.snackBarFn(); this.initSnackBarFn = true;};
        if($(".arrow_down") && !this.initArrowScrollFn){this.arrowScrollFn(); this.initArrowScrollFn = true;};
        if($(".floating_banner") && !this.initFloatingBannerMotionFn){this.floatingBannerMotionFn(); this.initFloatingBannerMotionFn = true;};
        remote.init();
        if($('.ie').length === 0 && $('.card_design_container').length > 0){//ex일때 인터렉션 실행 중지
            _this.cardDesignParallaxEffect.cardType2.init();
            $(window).resize(function(){
                _this.cardDesignParallaxEffect.cardType2.resize();
            });
        }
        // cardType2.init();
    },
    snackBarFn: function() {
        //스낵바 연회비 보기
        $(".member_fee .view_btn").off('click.member_fee_view').on('click.member_fee_view', function () {
            $('.fee_box').show();
            $('.member_fee').addClass('on')
        });
        $(".member_fee .btn_close_box").off('click.member_fee_close').on('click.member_fee_close', function () {
            $('.fee_box').hide();
            $('.member_fee').removeClass('on');
        });
        // 스낵바 나오는 타이밍 조정
        if($('.cont_wrap .apply_btn').size()){
            var eventTop = $('.cont_wrap .apply_btn').offset().top - 88;
            $(window).scroll(function(){
                var breadEvt = $(window).scrollTop() > eventTop;
                if(breadEvt === true){
                    $('.snack_bar').addClass('fixed');
                } else {
                    $('.snack_bar').removeClass('fixed');
                }
            });
        }
    },
    arrowScrollFn: function() {
        // arrow 버튼 클릭시 스크롤
        $(document).off('click.arrow_down').on('click.arrow_down' , '.arrow_down' , function(){
            var arrowTop = $('#cms_area').length > 0 ? $('#cms_area').offset().top - 112 : 0;
            var scrollTop = $(window).scrollTop();
            console.log(arrowTop , scrollTop);
            $('html , body').animate({scrollTop : arrowTop} , 300);
        });
    },
    floatingBannerMotionFn: function(){
            // 플러팅 배너 모션
        $(document).off('click.floating_banner_close').on('click.floating_banner_close', '.floating_banner .close_btn', function(){
            //$('.floating_banner').addClass('close');
            $('.floating_banner').animate({
                right: '-355px'
            }, 200, function (){// Animation complete
                $(this).css('display','none');
            })
        });

        //우측 플로팅 배너 고정
        if($('.floating_banner').length === 1){
            $(window).scroll(function(){
                var scrT = $(window).scrollTop();
                var scrEd = $(document).height() - $(window).height() - $('#footer').outerHeight();
                if(scrT >= scrEd){
                    per = (scrT - scrEd) + 22
                    $('.floating_banner').css({'bottom': per + 'px'});
                } else {
                    $('.floating_banner').css({'bottom': ''})
                }

            });
        }
    },
    cardDesignParallaxEffect : {
        cardType2 : {
            thisEventCallCheck : 0,
            designLen : $('.card_design_wrap ul li').length,
            scrollVal : 1000,
            eventSt :  null,
            pdTop : null,
            init: function(){
                var _this = this;
                _this.designLen = $('.card_design_wrap ul li').length;
                _this.event();
                _this.reload();
                console.log('인터렉션B');
            },
            event: function(){
                var _this = this;
                _this.eventHt =  $('.card_design_container').height() + _this.scrollVal;
                $('.card_design_container').css({'height' : _this.eventHt  + 'px' , 'opacity':'0'});
                _this.scalePt = _this.designLen > 7 ? 0.48 : 0.3;
                _this.scaleFt = _this.designLen > 7 ? 0.68 : 0.43;
                _this.transYft = _this.designLen > 7 ? 200 : 100;
                _this.lastscroll = 0;
                _this.scrollTop = $(window).scrollTop();
                _this.eventEd = $('.card_design_container').next('div').offset().top  - $(window).height() - 112; //2022-06-22 GGCN45 하단 영역 계산 수정(상단고정영역 값 48px + 64px)
                _this.eventEd1 = _this.eventEd;
                _this.eventEd2 = $(window).height() - 759;
                if(_this.thisEventCallCheck === 0){
                    _this.thisEventCallCheck ++;
                    console.log( "_this.thisEventCallCheck"  + _this.thisEventCallCheck );
                    window.addEventListener('scroll', function(){
                        _this.scrollTop = $(window).scrollTop();
                        cardBasicInteractionFn.cardDesignParallaxEffect.cardType2.eventAction()
                    });
                    // $(window).scroll(function(){
                    // });
                }
            },
            eventAction: function(){
                var _this = this;
                _this.eventSt =  $('.card_design_container').offset().top - $(window).height() + 256;
                if(_this.scrollTop < _this.eventSt){
                    $('.card_design_wrap ul').css({'transform':'scale(1) translateY(0)' , 'transition':'transform .0s'});
                    $('.card_design_wrap ul .p1_m_ctr_1ln').css({'transform':'scale(1)' , 'margin-top': '12px'});
                    $('.card_design_wrap').removeClass('fixed').css({'position':''});
                    $('.card_design_container').removeClass('start').css({'opacity':'0'});
                } else if(_this.scrollTop >= _this.eventSt && _this.scrollTop < _this.eventEd1){
                    $('.card_design_container').addClass('start').css({'opacity':'1' , 'transition':'opacity 0s'});
                    $('.card_design_container').next('div').css({'opacity':'0' , 'margin-top':'' , 'transition':'opacity .3s , margin .3s'});
                    if(_this.scrollTop >= $('.card_design_container').offset().top){
                        $('.card_design_wrap').addClass('fixed').css({'top': '' , 'bottom':''});
                        $('.card_design_wrap ul').css({'transform':'scale('+(1 - _this.scalePt)+') translateY(-'+_this.transYft+'px)' , 'transition':'transform .5s'});
                        $('.card_design_wrap ul .p1_m_ctr_1ln').css({'transform':'scale('+(1+_this.scaleFt)+')' , 'margin-top': 12 * (1+_this.scaleFt) + 'px' , 'transition':'transform .5s'});
                    } else {
                        $('.card_design_wrap').removeClass('fixed').css({'position':''});
                        $('.card_design_wrap ul').css({'transform':'scale(1) translateY(0)' , 'transition':'transform .5s'});
                        $('.card_design_wrap ul .p1_m_ctr_1ln').css({'transform':'scale(1)' , 'margin-top': '12px'});
                    }
                } else if(_this.scrollTop >= _this.eventEd1 && _this.scrollTop < _this.eventEd + _this.eventEd2) {
                    _this.otp = (_this.scrollTop - _this.eventEd1) / ((_this.eventEd + _this.eventEd2) - _this.eventEd1);
                    $('.card_design_wrap').addClass('fixed').css({'top': '' , 'bottom':''});
                    $('.card_design_wrap ul').css({'transform':'scale('+(1 - _this.scalePt)+') translateY(-'+_this.transYft+'px)' , 'transition':'transform .5s'});
                    $('.card_design_wrap ul .p1_m_ctr_1ln').css({'transform':'scale('+(1+_this.scaleFt)+')' , 'margin-top': 12 * (1+_this.scaleFt) + 'px', 'transition':'transform .5s'});
                    $('.card_design_container').addClass('start').css({'opacity':'1'});
                    $('.card_design_container').next('div').css({'margin-top': '', 'opacity':_this.otp * 0.7 , 'transition':'opacity .3s , margin .2s'});
                } else {
                    console.log("ewtrqwerqwerqwfeqaf");
                    $('.card_design_wrap').removeClass('fixed').css({'top': 'unset' , 'bottom':'0'});
                    $('.card_design_wrap ul').css({'transform':'scale('+(1 - _this.scalePt)+') translateY(-'+_this.transYft+'px)' , 'transition':'transform .5s'});
                    $('.card_design_wrap ul .p1_m_ctr_1ln').css({'transform':'scale('+(1+_this.scaleFt)+')' , 'margin-top': 12 * (1+_this.scaleFt) + 'px'});
                    $('.card_design_container').addClass('start').css({'opacity':'1'});
                    $('.card_design_container').next('div').css({'opacity':'1' ,'margin-top' : '0' , 'transition':'opacity .6s , margin .2s'});
                }

            },
            reload:function(){
                var _this = this;
                _this.eventAction();
            },
            resize:function(){
                var _this = this;
                _this.eventEd = $('.card_design_container').next('div').offset().top  - $(window).height() - 160;
                _this.eventEd1 = _this.eventEd;
                _this.eventEd2 = $(window).height() - 759;
                _this.pdTop = $(window).height() - $('.card_design_container').height() - 272;
                _this.eventAction();
            }
        }
    }
};

$(window).load(function () {
    cardBasicInteractionFn.init(); //window.load 후 카드상세 기본 인터렉션 호출
});
/* s : 2022-05-25 GGCN45 : window.onload 내부 FN 리팩토링 */
/* 2022-04-12 GGCN45 : 금융서비스 영역 공통화 처리 */
var $financeServiceArea =  '<!--from. card_detail.js --><div class="finance_service_inner_wrap">' +
                           '    <div class="finance_service_inner">' +
                           '        <p class="h3_b_lt" style="margin-left: 24px;">금융서비스</p>' +
                           '        <ul class="box_list_warp mt24">' +
                           '            <li>' +
                           '                <a href="/cpf/ma/CPFMA0101_01.hc">' +
                           '                    <p class="h3_b">목돈이<br>필요할 때</p>' + // 2022-12-21 Q20016 가이드 변경으로 폰트 색상 변경(class="fc_wht" 삭제)
                           '                    <p class="p1_b_1 mt12">' +
                           '                            <span class="textbico_small">장기카드대출(카드론)</span>' + // 2022-12-21 Q20016 금융배너 가이드 변경으로 폰트 색상 변경(class="fc_wht" 삭제)
                           '                    </p>' +
                           '                </a>' +
                           '            </li>' +
                           '            <li>' +
                           '                <a href="/cpf/ma/CPFMA0101_01.hc">' +
                           '                    <p class="h3_b">일상에서 현금이<br>' + // 2022-12-21 Q20016 금융배너 가이드 변경으로 폰트 색상 변경(class="fc_wht" 삭제)
                           '                            필요할 때' +
                           '                    </p>' +
                           '                    <p class="p1_b_lt_1ln mt12">' +
                           '                            <span class="textbico_small">단기카드대출(현금서비스)</span>' + // 2022-12-21 Q20016 금융배너 가이드 변경으로 폰트 색상 변경(class="fc_wht" 삭제)
                           '                    </p>' +
                           '                </a>' +
                           '            </li>' +
                           '            <li>' +
                           '                <a href="/cpf/ma/CPFMA0101_01.hc">' +
                           '                    <p class="h3_b">카드대금이<br>부담될 때</p>' + // 2022-12-21 Q20016 금융배너 가이드 변경으로 폰트 색상 변경(class="fc_wht" 삭제)
                           '                    <p class="p1_b_lt_1ln mt12">' +
                           '                            <span class="textbico_small">일부결제금액이월약정(리볼빙)</span>' + // 2022-12-21 Q20016 금융배너 가이드 변경으로 폰트 색상 변경(class="fc_wht" 삭제)
                           '                    </p>' +
                           '                </a>' +
                           '            </li>' +
                           '        </ul>' +
                           '    </div>' +
                           '</div>';

var cardDetailLoadedCallback = {
    init : function(){
        console.log("cardDetailLoadedCallback");
        appendPay1(); // 2023-03-03 GGU282 - 추가
        appendOnlineShopping1(); // 2023-03-06 GGU282 - 추가
        cardBasicInteractionFn.init(); //카드 상세 CMS 로드 후 카드상세 기본 인터렉션 호출
        if(Object.keys(cardDetailLoadedCallback.externalCallBackFn).length > 0){ // 외부에서 등록한 callback 함수가 있으면 카드상세 로드 후 실행.
            Object.keys(cardDetailLoadedCallback.externalCallBackFn).forEach(function(element, index){
                cardDetailLoadedCallback.externalCallBackFn[Object.keys(cardDetailLoadedCallback.externalCallBackFn)[index]]();
            });
        }
    },
    externalCallBackFn : {
        // test1 : function(){
        //     console.log('test1');
        // },
        // test2 : function(){
        //     console.log('test2');
        // },
    },
};

function appendPay1() { // 2023-03-03 GGU282 - 추가
    var tmpHtml =  '<!--from. card_detail.js -->' +
    '<p class="img"><img class="svg" alt="Apple Pay" src="/docfiles/resources/pc/images/detail/ico_applepay.svg"></p>' +
    '<h3 class="h3_b_ctr mt24">해당 상품은 Apple Pay 이용이 가능합니다.</h3>' +
    '<p class="p1_b_1 mt28"><a class="textbico_small_link fc_m_link" href="/cpu/ug/CPUUG4001_01.hc">Apple Pay 이용 안내 바로가기</a></p>';

    var $target = $('.pay1_area');
    if($target != null) {
        $target.html(tmpHtml);
    }
}

function appendOnlineShopping1() { // 2023-03-06 GGU282 - 추가
    var tmpHtml =  '<!--from. card_detail.js -->' +
    '<div class="item_tit">' +
    '    <em class="h3_b_lt">온라인 쇼핑</em>' +
    '    <p class="h0_b_lt_size40">온라인 쇼핑 시 추가 M포인트 적립</p>' +
    '</div>' +
    '<ul class="step_list">' +
    '    <li>' +
    '        <div class="img"><img class="svg" src="/docfiles/resources/mo/images/common/svg/ico_on_001.svg" alt=""></div>' +
    '        <p class="p1_b_ctr_2ln">현대카드<br>홈페이지 접속</p>' +
    '    </li>' +
    '    <li>' +
    '        <div class="img"><img class="svg" src="/docfiles/resources/mo/images/common/svg/ico_shop_002.svg" alt=""></div>' +
    '        <p class="p1_b_ctr_2ln">제휴 온라인쇼핑몰<br>쇼핑 및 결제</p>' +
    '    </li>' +
    '    <li>' +
    '        <div class="img"><img class="svg" src="/docfiles/resources/mo/images/common/svg/ico_pt_002.svg" alt=""></div>' +
    '        <p class="p1_b_ctr_2ln">M포인트<br>추가 적립</p>' +
    '    </li>' +
    '</ul>';

    var $target = $('.onlineshopping1_area');
    if($target != null) {
        $target.closest('.item_wrap').addClass('step_icon_list2');
        $target.html(tmpHtml);
    }
}