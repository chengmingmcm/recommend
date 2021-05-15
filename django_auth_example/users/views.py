import os

from django.shortcuts import redirect

from users.util.ItemBasedCF import *
from users.util.LFM import *
from users.util.UserBasedCF import *
from users.util.database_connect import *
from .forms import RegisterForm
from users.models import Resulttable, Insertposter


# 注册方法
def register(request):
    # 只有当请求为 POST 时，才表示用户提交了注册信息
    if request.method == 'POST':
        form = RegisterForm(request.POST)

        # 验证数据的合法性
        if form.is_valid():  # 如果提交数据合法，调用表单的 save 方法将用户数据保存到数据库
            form.save()

            # 注册成功，跳转回首页
            return redirect('/')
    else:
        pass
        # 请求不是 POST，表明用户正在访问注册页面，展示一个空的注册表单给用户
        form = RegisterForm()

    # 渲染模板
    # 如果用户正在访问注册页面，则渲染的是一个空的注册表单
    # 如果用户通过表单提交注册信息，但是数据验证不合法，则渲染的是一个带有错误信息的表单
    return render(request, '../templates/users/register.html', context={'form': form})


def index(request):
    return render(request, 'users/..//index.html')


def check(request):
    return render((request, 'users/..//index.html'))


# 用于展示用户评过的电影
def showmessage(request):
    usermovieid = []
    usermovietitle = []
    userid = request.GET.get('userIdd')
    # 查询数据库中的数据userID = 1001
    data = Resulttable.objects.filter(uersId=userid)
    for row in data:
        usermovieid.append(row.imdbId)

    try:
        conn = get_conn()
        cur = conn.cursor()
        for i in usermovieid:
            cur.execute('select * from moviegenre3 where imdbId = %s', i)
            rr = cur.fetchall()
            for imdbId, title, poster in rr:
                usermovietitle.append(title)
                print(title)

        # print(poster_result)
    finally:
        conn.close()
    return render(request, 'users/message.html', locals())


def recommend1(request):
    USERID = int(request.GET.get("userIdd")) + 1000
    read_mysql_to_csv('users/static/users_resulttable.csv', USERID)  # 追加数据，提高速率
    ratingfile = os.path.join('users/static', 'users_resulttable.csv')


    usercf = UserBasedCF()
    userid = str(USERID)  # 得到了当前用户的id 1001
    usercf.generate_dataset(ratingfile, 0.7)  # 划分训练集和测试集
    usercf.calc_user_sim()
    usercf.recommend(userid)  # 得到imdbId号


    ratio_range = [1, 2, 3, 4, 5]
    mae_rmse = []
    pre_rec = []
    for i in ratio_range:
        lfm = LFM(train_size=0.8, ratio=i)
        if (os.path.exists(model_path)):
            lfm.load()
        else:
            lfm.train()
            print("train over...")
            mae, rmse = lfm.validate()
            print('MAE:', mae, 'RMSE', rmse)  # rmse:均方根误差 mae:平均绝对误差
            mae_rmse.append((mae, rmse))
        pre, rec = lfm.evaluate()
        print('precision:', pre, 'recall:', rec)  # pre:精确度 rec
        # Precision 就是检索出来的条目中（比如：文档、网页等）有多少是准确的，Recall就是所有准确的条目有多少被检索出来了。
        pre_rec.append((pre, rec))
    print(mae_rmse, pre_rec)






    try:
        conn = get_conn()
        cur = conn.cursor()
        Insertposter.objects.filter(userId=USERID).delete() # 清空insertposter库，防止以前的数据产生影响
        for i in matrix:
            cur.execute('select * from moviegenre3 where imdbId = %s', i)
            rr = cur.fetchall()
            for imdbId, title, poster in rr:
                if (Insertposter.objects.filter(title=title)):
                    continue
                else:
                    Insertposter.objects.create(userId=USERID, title=title, poster=poster)

        # print(poster_result)
    finally:
        conn.close()
    results = Insertposter.objects.filter(userId=USERID)

    return render(request, 'users/movieRecommend.html', locals()    )
    # return render(request, 'users/..//index.html', locals())


# 用于展示推荐数据，调用计算函数


def recommend2(request):
    USERID = int(request.GET.get('userIdd')) + 1000
    # USERID = 1001
    # Insertposter.objects.filter(userId=USERID).delete()
    # selectMysql()
    read_mysql_to_csv2('users/static/users_resulttable2.csv', USERID)  # 追加数据，提高速率
    ratingfile2 = os.path.join('users/static', 'users_resulttable2.csv')
    itemcf = ItemBasedCF()
    # userid = '1001'
    userid = str(USERID)  # 得到了当前用户的id
    itemcf.generate_dataset(ratingfile2)
    itemcf.calc_movie_sim()
    itemcf.recommend(userid)  # 得到imdbId号

    # 先删除所有数据

    try:
        conn = get_conn()
        cur = conn.cursor()
        # Insertposter.objects.filter(userId=USERID).delete()
        for i in matrix2:
            cur.execute('select * from moviegenre3 where imdbId = %s', i)
            rr = cur.fetchall()
            for imdbId, title, poster in rr:
                # print(value)         #value才是真正的海报链接
                if (Insertposter.objects.filter(title=title)):
                    continue
                else:
                    Insertposter.objects.create(userId=USERID, title=title, poster=poster)

        # print(poster_result)
    finally:
        conn.close()
        results = Insertposter.objects.filter(userId=USERID)  # 从这里传递给html= Insertposter.objects.all()

    return render(request, 'users/movieRecommend2.html', locals())
    # return HttpResponseRedirect('movieRecommend.html', locals())


#
if __name__ == '__main__':
    ratingfile2 = os.path.join('static', 'users_resulttable.csv')  # 一共671个用户

    usercf = UserBasedCF()
    userId = '1'
    # usercf.initial_dataset(ratingfile1)
    usercf.generate_dataset(ratingfile2)
    usercf.calc_user_sim()
    # usercf.evaluate()
    usercf.recommend(userId)
    # 给用户推荐10部电影  输出的是‘movieId’,兴趣度
